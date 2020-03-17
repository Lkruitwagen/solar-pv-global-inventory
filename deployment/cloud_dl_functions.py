def S2Infer1(dltile, src_product_id, dest_product_id, fc_id, sdate,edate, run_ii):
    import tensorflow as tf
    import numpy as np
    import keras, json, time, datetime, pyproj
    from shapely import geometry
    from shapely.ops import transform
    from functools import partial
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    from skimage import measure
    from pyproj import Proj
    from scipy import ndimage, spatial
    from scipy.signal import fftconvolve
    from PIL import Image, ImageDraw

    import descarteslabs as dl

    #load all the clients
    catalog_client = dl.Catalog()
    raster_client = dl.Raster()
    metadata_client = dl.Metadata()
    storage_client = dl.Storage()

    tic = time.time()


    def watershed_labels(inp_arr):
        distance = ndimage.distance_transform_edt(inp_arr)
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((1, 1)),
                            labels=inp_arr)
        markers = ndimage.label(local_maxi)[0]
        labels = watershed(-distance,
                           markers, mask=inp_arr)

        return labels

    def gaussian_blur(in_array, size):
        # expand in_array to fit edge of kernel
        padded_array = np.pad(in_array, size, 'symmetric')
        # build kernel
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
        g = (g / g.sum()).astype(in_array.dtype)
        # do the Gaussian blur
        return fftconvolve(padded_array, g, mode='valid')

    def vectorise_2(arr, tile_size, disp=2):

        polys = []

        contours = measure.find_contours(np.pad(arr,disp,'constant', constant_values=(0)), 0.8)
        for c in contours:
            #print ('c shape',c.shape)
            #print ('assrt', c[:,:]==c[1])

            c = (c-disp).clip(0.,float(tile_size))
            poly = geometry.Polygon(c)
            poly = poly.simplify(0.1)
            polys.append(poly)
        return polys

    def pixXY2lonlat(pt,dt):
        X = pt[0]
        Y = pt[1]
        lon = X*dt[1]+dt[0]+Y*dt[2]
        #print Y
        lat = Y*(dt[5]-dt[2]*dt[4]/dt[1])+dt[3]+(lon-dt[0])*dt[4]/dt[1]
        #print X
        return [lon,lat]

    def pixvec2rast(polys, tilesize):

        blank = np.zeros((tilesize, tilesize))
        im = Image.fromarray(blank, mode='L')
        draw = ImageDraw.Draw(im)

        for p in polys:
            xs, ys = p.exterior.xy
            draw.polygon(list(zip(xs,ys)), fill=255)

        return np.array(im)





    #fetch the scene metadata

    scenes = metadata_client.search(src_product_id, geom=raster_client.dltile(dltile['properties']['key']), start_datetime=sdate,  end_datetime=edate, cloud_fraction=0.2, limit=15)['features']

    #get the least cloudy one
    #scenes = sorted(scenes, key=lambda k: k.properties.cloud_fraction, reverse=False)
    scenes = sorted(scenes, key=lambda k: k.properties.acquired, reverse=True) #do most recent scene first

    print ('n scenes: ', len(scenes), 'toc',time.time()-tic)

    if len(scenes)==0:
        scenes = metadata_client.search(src_product_id, geom=raster_client.dltile(dltile['properties']['key']), start_datetime='2018-07-01',  end_datetime='2018-09-30', cloud_fraction=0.2, limit=15)['features']

        #get the least cloudy one
        #scenes = sorted(scenes, key=lambda k: k.properties.cloud_fraction, reverse=False)
        scenes = sorted(scenes, key=lambda k: k.properties.acquired, reverse=True) #do most recent scene first

        print ('n new scenes: ', len(scenes))


    #AAALLL THE BANDS
    bands = ['red', 'green', 'blue', 'nir', 'red-edge','red-edge-2', 'red-edge-3', 'red-edge-4', 'swir1','swir2','water-vapor','cirrus','coastal-aerosol','alpha']


    ### make the tile array
    tile_arr = np.zeros((len(scenes),dltile['properties']['tilesize'],dltile['properties']['tilesize'],len(bands)),dtype=np.float32)

    scene_ind = 0
    fill_frac = 0.

    for ii_s, s in enumerate(scenes):
        new_arr, meta = raster_client.ndarray(
            s['id'], bands=bands, scales=[[0, 10000]] * 14, ot='UInt16', dltile=dltile['properties']['key'], processing_level='surface'
        )

        tile_arr[ii_s, :,:,:] = new_arr
        print ('collected iis:',ii_s, 'toc',time.time()-tic)
    new_arr = None



    ### fetch and rebuild the keras model
    json_str = storage_client.get('model_json', storage_type='data')

    model = keras.models.model_from_json(json_str)

    model_weight_shapes = dict(json.loads(storage_client.get('model_weight_shapes',storage_type='data')))

    rebuild_weights = []

    for ii in range(len(model_weight_shapes.keys())):
        w = np.frombuffer(storage_client.get('model_weights_'+str(ii), storage_type='data'), dtype='float32').reshape(model_weight_shapes[str(ii)])

        rebuild_weights.append(w)

    model.set_weights(rebuild_weights)
    print ('loaded model toc',time.time()-tic)

    #cleanup, save mem
    rebuild_weights = None
    json_str = None
    model_weight_shapes = None



    ### Iterate over the tile for the predictions
    large_prediction = np.zeros((len(scenes),tile_arr.shape[1],tile_arr.shape[2]), dtype=np.float32)
    print ('large_pred shape', large_prediction.shape)

    for ii_s in range(len(scenes)):

        for ii in range(int(dltile['properties']['tilesize']/200.)):
            for jj in range(int(dltile['properties']['tilesize']/200.)):
                inp_arr = tile_arr[np.newaxis,ii_s,ii*200:(ii+1)*200,jj*200:(jj+1)*200,:]
                ###  get prediction from model
                pred = np.array(model.predict((inp_arr/255.).clip(0.,1.)))[0,:,:,:]
                #pred = np.argmax(pred,2)
                large_prediction[ii_s,ii*200:(ii+1)*200,jj*200:(jj+1)*200] = (pred[:,:,1]-pred[:,:,0]).clip(0.,1.)
        print ('pred iis', ii_s, 'toc', time.time()-tic)

    #clean up, save mem
    model = None
    pred = None
    inp_arr = None
    tile_arr = tile_arr[:,:,:,-1] ### make tile_arr just the alpha mask to reduce peak memory usage


    stack_sum = np.sum(large_prediction>0.1, axis=0)
    stack_max = np.amax(large_prediction, axis=0)
    print ('stack_sum shape', stack_sum.shape)
    print ('stack_max shape', stack_max.shape)


    ### QC hard filters ... might remove some positives... probably want
    #QC_mask = np.zeros((dltile['properties']['tilesize'], dltile['properties']['tilesize']))
    #QC_mask[tile_arr[:,:,8]==0] = 1
    #QC_mask[tile_arr[:,:,9]==0] = 1
    #large_prediction[QC_mask>0] = 0.



    ### Vectorise Prediction
    #gauss_arr = gaussian_blur(stack_sum>0,0)
    #print (gauss_arr.shape)

    #labels = watershed_labels((gauss_arr>0.1))

    labels = watershed_labels((stack_sum>0.1)) #but maybe don't need the gaussian blur at all?


    unique_labels = np.unique(labels)

    unique_labels = np.delete(unique_labels, np.argwhere(unique_labels==0))

    polys = []
    for ul in unique_labels:

        v2 = vectorise_2(labels==ul, dltile['properties']['tilesize'],5)
        for v in v2:
            polys.append(v)


    ### Prepare geotransortm
    dt = [1, 10, 0, 1, 0, -10]
    dt[0] = dltile['properties']['outputBounds'][0]
    dt[3] = dltile['properties']['outputBounds'][3]

    print (dt)
    print (dltile['properties']['geotrans'])
    dt = dltile['properties']['geotrans']
    print('vectorised', 'toc', time.time()-tic)


    ### Change all the coords
    outp_polys = []

    fc = dl.vectors.FeatureCollection(fc_id)

    features = []

    WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    tile_srs = dltile['properties']['proj4']

    utm_proj = pyproj.Proj(tile_srs)
    wgs_proj = pyproj.Proj(WGS84)

    projection_func = partial(pyproj.transform, utm_proj, wgs_proj)

    gauss_ann = np.zeros(large_prediction.shape,dtype=np.float32)

    for ii_s in range(len(scenes)):
        gauss_ann[ii_s,:,:] = gaussian_blur(large_prediction[ii_s,:,:]>=0.1,5)

    for ii_p,poly in enumerate(polys):

        if poly.area>0.1:


            xs, ys = poly.exterior.xy


            utm_pts = []

            for ii in range(len(xs)):
                utm_pts.append((pixXY2lonlat([ys[ii],xs[ii]],dt)))

            utm_poly = geometry.Polygon(utm_pts)

            print('Poly', ii_p,'poly to utm', 'toc', time.time()-tic)
            #print ('utm poly',utm_poly)

            ll_poly = transform(projection_func, utm_poly)

            outp_polys.append(ll_poly)


            annotation = pixvec2rast([poly], dltile['properties']['tilesize']).T
            print ('pix poly area', poly.area, 'annotation sum',(annotation>0).sum())


            properties = {}
            print('Poly', ii_p,'poly to latlon', 'toc', time.time()-tic)


            for ii_s, s in enumerate(scenes):

                ### scene tile_arr[ii_s, :,:,:] = new_arr

                fill_portion = ((annotation>0) * (tile_arr[ii_s,:,:]>0)).sum()/(annotation>0).sum() ## tile_arr is now just the alpha mask


                if fill_portion>0.8:



                    #print('Poly', ii_p,'gauss blur', 'toc', time.time()-tic)
                    TP = ((gauss_ann[ii_s,:,:]>=0.1) * (annotation>0))
                    FN = ((gauss_ann[ii_s,:,:]<0.1) * (annotation>0))
                    P = TP.sum()/float((TP.sum()+FN.sum()))
                    #print('Poly', ii_p,'TP, FP, P', 'toc', time.time()-tic)
                    poly_mean = np.mean(large_prediction[ii_s,annotation>0])
                    #print('Poly', ii_p,'mean reduce', 'toc', time.time()-tic)
                    print ('mean',poly_mean, 'P',P,'fill portion: ', fill_portion, 'calc toc',time.time()-tic)
                    properties[s.id+':P']=float(P)
                    properties[s.id+':M']=float(poly_mean)
                else:
                    properties[s.id] = 'no coverage'
                    print ('no coverage','fill portion: ', fill_portion, 'calc toc',time.time()-tic)


            if not (ll_poly.buffer(0).is_empty):
                properties['dltile']=dltile['properties']['key']
                properties['primary_id']=dltile['properties']['key']+'_'+str(ii_p)
                features.append(
                    dl.vectors.Feature(
                        geometry=ll_poly,
                        properties=properties
                    ))
                #print ('adding poly')
                print('Poly', ii_p,'create feature', 'toc', time.time()-tic)
            else:
                print ('empty poly')

            print ('len features', len(features))


    if len(features)>0:

        index= 0
        step=100
        while (index*step)<len(features):
            fc.add(features[index*step:min((index+1)*step,len(features))])
            index+=1

    large_prediction = None

    ### prepare image

    #image = np.full(tile_arr.shape[1:-1], stack_sum/15.*255, dtype=np.uint8)
    image = np.full(stack_max.shape, stack_max*255, dtype=np.uint8)
    print('image shape ', image.shape)

    image_id = ':'.join([str(datetime.date.today()),src_product_id, dltile['properties']['key'].replace(':', '_'),str(run_ii)])


    print ('uploading image...')

    cat_return = catalog_client.upload_ndarray(image, dest_product_id, image_id, raster_meta=meta, overviews=[5,10], overview_resampler='average')

    print ('catalog return')
    print (cat_return)

    output = {}
    output['dltile'] = dltile['properties']['key']
    output['image_id'] = image_id
    output['len_scenes'] = len(scenes)
    output['pred_>0_pix'] = int((image>0).sum())
    output['n_features']=len(features)
    #output['outp_features']=features


    #assert (dltile['properties']['key'] in scene_check_ids)

    return json.dumps(output) #scene_check_ids
    #return large_prediction

def S2Infer2(storage_key, dl_ft, src_product_id, dest_fc_id, storage_flag):
    import tensorflow as tf
    import numpy as np
    import keras, json, time, datetime, itertools, pyproj
    from shapely import geometry
    from shapely.ops import transform
    from functools import partial
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    from skimage import measure
    from pyproj import Proj
    from scipy import ndimage, spatial
    from scipy.signal import fftconvolve
    from PIL import Image, ImageDraw

    import descarteslabs as dl
    from descarteslabs.vectors import properties as dl_p

    #load all the clients
    catalog_client = dl.Catalog()
    raster_client = dl.Raster()
    metadata_client = dl.Metadata()
    storage_client = dl.Storage()


    def watershed_labels(inp_arr):
        distance = ndimage.distance_transform_edt(inp_arr)
        local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((1, 1)),
                            labels=inp_arr)
        markers = ndimage.label(local_maxi)[0]
        labels = watershed(-distance,
                           markers, mask=inp_arr)

        return labels

    def gaussian_blur(in_array, size):
        # expand in_array to fit edge of kernel
        padded_array = np.pad(in_array, size, 'symmetric')
        # build kernel
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
        g = (g / g.sum()).astype(in_array.dtype)
        # do the Gaussian blur
        return fftconvolve(padded_array, g, mode='valid')

    def vectorise_2(arr, tile_size, disp=2):

        polys = []

        contours = measure.find_contours(np.pad(arr,disp,'constant', constant_values=(0)), 0.8)
        for c in contours:
            #print ('c shape',c.shape)
            #print ('assrt', c[:,:]==c[1])

            c = (c-disp).clip(0.,float(tile_size))
            poly = geometry.Polygon(c)
            poly = poly.simplify(0.1)
            polys.append(poly)
        return polys

    def pixXY2lonlat(pt,dt):
        X = pt[0]
        Y = pt[1]
        lon = X*dt[1]+dt[0]+Y*dt[2]
        #print Y
        lat = Y*(dt[5]-dt[2]*dt[4]/dt[1])+dt[3]+(lon-dt[0])*dt[4]/dt[1]
        #print X
        return [lon,lat]

    def lonlat2pixXY(pt,dt):
        lon = pt[0]
        lat = pt[1]
        Y = (lat-dt[3]-dt[4]/dt[1]*(lon-dt[0]))/(dt[5]-dt[2]*dt[4]/dt[1])
        #print Y
        X = (lon-dt[0]-Y*dt[2])/dt[1]
        #print (lon,dt[0],X)
        #print X
        return [int(X),int(Y)]

    def rast2pixvec(input_prediction, tilesize,threshold, gauss_kernel):

        ### INPUT: np raster, threshold for greater than 'positive'
        ### OUTPUT: list of shapely polygons in pixel coordinates

        gauss_arr = gaussian_blur(input_prediction,gauss_kernel)
        #plt.imshow(gauss_arr)
        #plt.show()
        #print (gauss_arr.shape)

        labels = watershed_labels((gauss_arr>threshold))
        unique_labels = np.unique(labels)

        unique_labels = np.delete(unique_labels, np.argwhere(unique_labels==0))

        polys = []
        for ul in unique_labels:

            v2 = vectorise_2(labels==ul, tilesize,5)
            for v in v2:
                polys.append(v)

        return polys

    def pixvec2llvec(pixpolys,dt, ll2utm):
        out_polys = []

        for poly in pixpolys:
            xs, ys = poly.exterior.xy

            utm_pts = []
            ll_pts = []
            for ii in range(len(xs)):
                utm_pts.append((pixXY2lonlat([ys[ii],xs[ii]],dt)))


            for pt in utm_pts:
                ll_pts.append(list(ll2utm(*pt, inverse=True)))

            out_polys.append(geometry.Polygon(ll_pts))


            if not (geometry.Polygon(ll_pts).buffer(0).is_empty):
                out_polys.append(geometry.Polygon(ll_pts).buffer(0))
            else:
                print ('empty poly')

        return out_polys

    def flatten_polys(polys):
        polys_flattened=[]
        for pp in polys:
            if pp.type=='MultiPolygon':
                polys_flattened+=list(pp)
            elif pp.type=='Polygon':
                polys_flattened+=[pp]
            else:
                print ('not poly',pp.type)
        return polys_flattened

    def llvec2pixvec(llpolys,dt, ll2utm):
        out_polys = []

        for p in flatten_polys(llpolys):
            #print (geometry.Polygon([ll2utm(*c) for c in geometry.mapping(p)['coordinates'][0]]))
            print ('p_type',p.type)
            if p.type=='MultiPolygon':
                print ('multip', len(geometry.mapping(p)['coordinates'][0]))
                p_utm = geometry.Polygon([ll2utm(*c) for c in geometry.mapping(p)['coordinates'][0][0]])
                ###
            elif p.type=='Polygon':
                #print ([c for c in geometry.mapping(p)['coordinates'][0]])
                p_utm = geometry.Polygon([ll2utm(*c) for c in geometry.mapping(p)['coordinates'][0]])
            #print (p_utm)
            else:
                print ('geometry_collection')
                return out_polys

            for coords_utm in geometry.mapping(p_utm)['coordinates']:
                coords_utm = np.squeeze(np.array(coords_utm)).tolist()

                coords_pix = [lonlat2pixXY(c,dt) for c in coords_utm]
                #print (coords_pix)



                if not (geometry.Polygon(coords_pix).buffer(0).is_empty):
                    if (geometry.Polygon(coords_pix).buffer(0).type=='MultiPolygon'):
                        out_polys+=[p for p in geometry.Polygon(coords_pix).buffer(0)]
                    else:
                        out_polys.append(geometry.Polygon(coords_pix).buffer(0))
                else:
                    print ('empty poly')

        return out_polys

    def pixvec2rast(intersect_polys_pix, tilesize):

        blank = np.zeros((tilesize, tilesize))
        im = Image.fromarray(blank, mode='L')
        draw = ImageDraw.Draw(im)

        for p in intersect_polys_pix:
            xs, ys = p.exterior.xy
            draw.polygon(list(zip(xs,ys)), fill=255)

        return np.array(im)


    def get_stack(tile, poly):
        ### get scenes for dltile
        print ('doing tile: ',tile['properties']['key'])
        tic = time.time()
        scene_ind = 0
        scenes_ret = metadata_client.search('sentinel-2:L1C', geom=tile, start_datetime="2016-01-01",  end_datetime="2018-09-30", cloud_fraction=0.2, limit=300)
        scenes = scenes_ret['features']
        #token = scenes_ret['properties']['continuation_token']
        #print (len(scenes), token)
        #while token is not None:
        #    scenes_ret = metadata_client.search(continuation_token=token)
        #    scenes += scenes_ret['features']
        #    token = scenes_ret['properties']['continuation_token']

        print ('n scenes', len(scenes))

        scenes = sorted(scenes, key=lambda k: k.properties.acquired, reverse=True) ## <-- HERE

        #for s in scenes:
        #   print (s.properties.acquired, s.id)

        #print ('zones', scene['properties']['projcs'].split(' ')[-1], tile['properties']['zone'])
        ll2utm = Proj(proj='utm', zone=str(tile['properties']['zone']), ellps='WGS84')

        #get geometry transformations
        dt = scenes[0]['properties']['geotrans']
        print ('scenes dt',dt)
        dt[0] = ll2utm(*tile['geometry']['coordinates'][0][0])[0]
        dt[3] = ll2utm(*tile['geometry']['coordinates'][0][2])[1]
        print ('adj dt',dt)
        dt = tile['properties']['geotrans']
        print ('tile trans',tile['properties']['geotrans'])

        #print(poly)
        #print (llvec2pixvec([poly],dt,ll2utm))
        #annotation = #pixvec2rast(llvec2pixvec([poly],dt,ll2utm), tile['properties']['tilesize'])
        WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        tile_srs = tile['properties']['proj4']

        utm_proj = pyproj.Proj(tile_srs)
        wgs_proj = pyproj.Proj(WGS84)

        projection_func = partial(pyproj.transform, wgs_proj, utm_proj)
        utm_polys = flatten_polys([transform(projection_func, poly)])

        blank = np.zeros((tile['properties']['tilesize'], tile['properties']['tilesize']))
        im = Image.fromarray(blank, mode='L')
        draw = ImageDraw.Draw(im)

        for utm_poly in utm_polys:

            pix_poly = geometry.Polygon([lonlat2pixXY(c,dt) for c in list(utm_poly.exterior.coords)])
            xs,ys = pix_poly.exterior.xy
            #print (int(np.round(ft['properties']['prediction']*255,0)))
            draw.polygon(list(zip(xs,ys)),fill=255)

            print ('interiors:', len(utm_poly.interiors))
            if len(utm_poly.interiors)>0:
                for int_ring in utm_poly.interiors:
                    pix_poly = geometry.Polygon([lonlat2pixXY(c,dt) for c in list(int_ring.coords)])
                    xs,ys = pix_poly.exterior.xy
                    #print (int(np.round(ft['properties']['prediction']*255,0)))
                    draw.polygon(list(zip(xs,ys)),fill=0)

        annotation = np.array(im).astype('uint8')


        ### get stack
        bands = ['red', 'green', 'blue', 'nir', 'red-edge','red-edge-2', 'red-edge-3', 'red-edge-4', 'swir1','swir2','water-vapor','cirrus','coastal-aerosol','alpha']

        all_precisions = {}

        for s in scenes:
            print ('doing scene: ',s.id, 'toc:', (time.time()-tic))
            tile_arr, meta = raster_client.ndarray(
                s.id, bands=bands, scales=[[0, 10000]] * 14, ot='UInt16', dltile=tile['properties']['key'], processing_level='surface'
            )
            pred = np.squeeze(model.predict((tile_arr[np.newaxis, ...]/255.).clip(0.,1.)))
            pred = (pred[:,:,1]-pred[:,:,0]).clip(0.,1.)


            #fig, axs = plt.subplots(1,4,figsize=(32,8))
            #axs[0].imshow((tile_arr[:,:,0:3]/255.*2.5).clip(0.,1.))
            #axs[1].imshow(annotation)
            #axs[2].imshow(pred)
            #plt.show()

            gauss_pred = gaussian_blur(pred>=0.1,0)

            TP = (gauss_pred>=0.1) * annotation>0
            FN = (gauss_pred<0.1) * annotation>0
            P = TP.sum()/(TP.sum()+FN.sum())
            poly_mean = np.mean(pred[annotation>0])
            print ('done scene: ',s.id, 'toc:', (time.time()-tic), 'P:',P,'M:',poly_mean)

            all_precisions[s.id] = {'TP':TP.sum(),'FN':FN.sum(), 'P':P,'M':poly_mean}

        return all_precisions


        ### get the tile_arr, benchmark to date,process through the model, get percision, take the fit \
        #llvec2pixvec(poly,dt, ll2utm)
        #pixvec2rast(intersect_polys_pix, tilesize):


    tic = time.time()
    print ('tic',tic)

    if storage_flag:
        ft_load = json.loads(storage_client.get(storage_key,storage_type='data'))
    else:
        ft_load = json.loads(dl_ft)

    ft_load = dl.vectors.Feature(
                    geometry=ft_load['geometry'],
                    properties=ft_load['properties']
                )

    print ('ft_load',ft_load)
    print ('toc', time.time()-tic)
    poly = geometry.shape(ft_load.geometry)

    print ('n holes',len(poly.interiors))

    print ('loaded feature:')
    print (ft_load)
    print ('toc', time.time()-tic)

    bbox = geometry.box(*poly.bounds)

    tiles = raster_client.dltiles_from_shape(10, 200, 0,geometry.mapping(bbox))
    #print (tiles['features'][0])
    print ('n_tiles: ',len(tiles['features']))

    ## load model
    ### fetch and rebuild the keras model
    json_str = storage_client.get('model_json', storage_type='data')

    model = keras.models.model_from_json(json_str)

    model_weight_shapes = dict(json.loads(storage_client.get('model_weight_shapes',storage_type='data')))

    rebuild_weights = []

    for ii in range(len(model_weight_shapes.keys())):
        w = np.frombuffer(storage_client.get('model_weights_'+str(ii), storage_type='data'), dtype='float32').reshape(model_weight_shapes[str(ii)])

        rebuild_weights.append(w)

    model.set_weights(rebuild_weights)
    #####
    print ('load model toc:',time.time()-tic)


    all_results = {}
    for ii_t,tile in enumerate(tiles['features']):
        #fig, axs = plt.subplots(1,1,figsize=(8,8))
        #xs, ys = geometry.Polygon(tile['geometry']['coordinates'][0]).exterior.xy
        #xs_p, ys_p = p.intersection(geometry.Polygon(tile['geometry']['coordinates'][0])).exterior.xy
        #axs.plot(xs, ys, 'b')
        #axs.plot(xs_p, ys_p, 'r')
        #plt.show()
        all_results[tile['properties']['key']] = get_stack(tile,poly.intersection(geometry.Polygon(tile['geometry']['coordinates'][0])))
        print ('tile toc', ii_t, time.time()-tic)

    print ('all results')
    for k,v in all_results.items():
        print (k,v)
    ### get all all scenes:
    all_scene_ids = list(set(list(itertools.chain(*[all_results[k].keys() for k in all_results.keys()]))))
    print ('all_scene_ids')
    print (all_scene_ids)

    for s_id in all_scene_ids:
        ### check that they're in all results
        check = np.array([s_id in all_results[k].keys() for k in all_results.keys()])
        #print ('s_di',s_id,'check',check)

        if np.prod(check)==1:
            TPs = [all_results[k][s_id]['TP'] for k in all_results.keys()]
            FNs = [all_results[k][s_id]['FN'] for k in all_results.keys()]
            P = sum(TPs)/float((sum(TPs)+sum(FNs)))
            print ('s_id',s_id)
            print ([all_results[k][s_id]['TP'] for k in all_results.keys()])
            print ([all_results[k][s_id]['M'] for k in all_results.keys()])
            sum_means = [(all_results[k][s_id]['TP']*all_results[k][s_id]['M']) for k in all_results.keys()]
            sum_means = [sm for sm in sum_means if not np.isnan(sm)]
            blended_mean = sum(sum_means)/sum(TPs)
            print ('blend mean',blended_mean)
            ft_load.properties[s_id+':P'] = P
            ft_load.properties[s_id+':M'] = blended_mean
    print ('get Ps toc', time.time()-tic)
    print ('props')
    print (ft_load.properties)




    ### group all the months together
    P_keys = []
    for k,v in ft_load.properties.items():
        if k.split(':')[0]=='sentinel-2':
            P_keys.append(k)
    P_key_dates = [k.split(':')[2][0:10] for k in P_keys]

    yyyymms = list(set([k[0:7]for k in P_key_dates]))

    properties_upload = {}
    for ym in yyyymms:
        properties_upload[ym] = {}

    for k in P_keys:
        ym = k.split(':')[2][0:7]
        dd = k.split(':')[2][8:10]
        if ((isinstance(ft_load.properties[k],float)) and (dd not in properties_upload[ym].keys())):
            properties_upload[ym][dd] = {}

        if ((isinstance(ft_load.properties[k],float)) and (k.split(':')[-1]=='P')):
            properties_upload[ym][dd]['P'] = "{0:.3f}".format(round(ft_load.properties[k],3))
        elif ((isinstance(ft_load.properties[k],float)) and (k.split(':')[-1]=='M')):
            properties_upload[ym][dd]['M'] = "{0:.3f}".format(round(ft_load.properties[k],3))
        #else:
        #    properties_upload[ym][dd] = ft_load['properties'][k]
    print ('handle properties toc', time.time()-tic)

    for k1 in properties_upload.keys():
        for k2,v2 in properties_upload[k1].items():
            properties_upload[k1][k2]=[v2['P'],v2['M']]


    for k,v in properties_upload.items():
        properties_upload[k] = json.dumps(v)

    for k,v in properties_upload.items():
        properties_upload[k] = v.replace('"','').replace(' ','')
    print ('properties for upload:')

    print (properties_upload)
    properties_upload['dltile']=ft_load.properties['dltile']
    properties_upload['primary_id'] = ft_load.properties['primary_id']
    properties_upload['prediction']=ft_load.properties['prediction']


    fc_dest = dl.vectors.FeatureCollection(dest_fc_id)
    fc_dest.add([
        dl.vectors.Feature(
                    geometry=geometry.polygon.orient(poly),
                    properties=properties_upload
                )
        ])

    return 0

def S2RNN1(dltile, src_vector_id, dest_vector_id, dest_product_id, push_rast = True):
    import descarteslabs as dl
    import tensorflow as tf
    import numpy as np
    import keras, json, time, datetime, math, geojson, pyproj

    from shapely import geometry
    from shapely.ops import transform
    from functools import partial
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    from skimage import measure
    from pyproj import Proj
    from scipy import ndimage, spatial
    from scipy.signal import fftconvolve
    from PIL import Image, ImageDraw

    #load all the clients
    catalog_client = dl.Catalog()
    raster_client = dl.Raster()
    metadata_client = dl.Metadata()
    storage_client = dl.Storage()

    tic = time.time()

    ### def some shit
    date_format = "%Y-%m-%d"
    baseline = datetime.datetime.strptime('2016-01-01', date_format)

    def lonlat2pixXY(pt,dt):
        lon = pt[0]
        lat = pt[1]
        Y = (lat-dt[3]-dt[4]/dt[1]*(lon-dt[0]))/(dt[5]-dt[2]*dt[4]/dt[1])
        #print Y
        X = (lon-dt[0]-Y*dt[2])/dt[1]
        #print (lon,dt[0],X)
        #print X
        return [int(X),int(Y)]

    def V_inv(point1, point2, miles=False):

        # WGS 84
        a = 6378137  # meters
        f = 1 / 298.257223563
        b = 6356752.314245  # meters; b = (1 - f)a

        MILES_PER_KILOMETER = 0.621371

        MAX_ITERATIONS = 200
        CONVERGENCE_THRESHOLD = 1e-12  # .000,000,000,001
        """
        Vincenty's formula (inverse method) to calculate the distance (in
        kilometers or miles) between two points on the surface of a spheroid

        Doctests:
        >>> vincenty((0.0, 0.0), (0.0, 0.0))  # coincident points
        0.0
        >>> vincenty((0.0, 0.0), (0.0, 1.0))
        111.319491
        >>> vincenty((0.0, 0.0), (1.0, 0.0))
        110.574389
        >>> vincenty((0.0, 0.0), (0.5, 179.5))  # slow convergence
        19936.288579
        >>> vincenty((0.0, 0.0), (0.5, 179.7))  # failure to converge
        >>> boston = (42.3541165, -71.0693514)
        >>> newyork = (40.7791472, -73.9680804)
        >>> vincenty(boston, newyork)
        298.396057
        >>> vincenty(boston, newyork, miles=True)
        185.414657
        """

        # short-circuit coincident points
        if point1[0] == point2[0] and point1[1] == point2[1]:
            return 0.0,0,0

        U1 = math.atan((1 - f) * math.tan(math.radians(point1[0])))
        U2 = math.atan((1 - f) * math.tan(math.radians(point2[0])))
        L = math.radians(point2[1] - point1[1])
        Lambda = L

        sinU1 = math.sin(U1)
        cosU1 = math.cos(U1)
        sinU2 = math.sin(U2)
        cosU2 = math.cos(U2)

        for iteration in range(MAX_ITERATIONS):
            sinLambda = math.sin(Lambda)
            cosLambda = math.cos(Lambda)
            sinSigma = math.sqrt((cosU2 * sinLambda) ** 2 +
                                 (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
            if sinSigma == 0:
                return 0.0  # coincident points
            cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
            sigma = math.atan2(sinSigma, cosSigma)
            sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
            cosSqAlpha = 1 - sinAlpha ** 2
            try:
                cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
            except ZeroDivisionError:
                cos2SigmaM = 0
            C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
            LambdaPrev = Lambda
            Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                                   (cos2SigmaM + C * cosSigma *
                                                    (-1 + 2 * cos2SigmaM ** 2)))
            if abs(Lambda - LambdaPrev) < CONVERGENCE_THRESHOLD:
                break  # successful convergence
        else:
            return None  # failure to converge

        uSq = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
        A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
        B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
        deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma *
                     (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM *
                     (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
        s = b * A * (sigma - deltaSigma)


        num = (math.cos(U2)*math.sin(Lambda))
        den = (math.cos(U1)*math.sin(U2)-math.sin(U1)*math.cos(U2)*math.cos(Lambda))

        #print 'num',num
        #print 'den',den
        alpha1 = math.atan2(num,den)

        if alpha1<0:
            alpha1+=2*math.pi



        num = (math.cos(U1)*math.sin(Lambda))
        den = (-1.0*math.sin(U1)*math.cos(U2)+math.cos(U1)*math.sin(U2)*math.cos(Lambda))
        #print 'num',num
        #print 'den',den
        alpha2 = math.atan2(num,den)

        if alpha2<0:
            alpha2+=2*math.pi


        s /= 1000  # meters to kilometers
        if miles:
            s *= MILES_PER_KILOMETER  # kilometers to miles

        return round(s, 6), math.degrees(alpha1), math.degrees(alpha2)

    def V_dir(point1, s, alpha1,miles=False):
        #print 'v_dir'
            # WGS 84
        a = 6378137  # meters
        f = 1 / 298.257223563
        b = 6356752.314245  # meters; b = (1 - f)a

        MILES_PER_KILOMETER = 0.621371

        MAX_ITERATIONS = 200
        CONVERGENCE_THRESHOLD = 1e-12  # .000,000,000,001
        """
        Vincenty's formula (inverse method) to calculate the distance (in
        kilometers or miles) between two points on the surface of a spheroid

        Doctests:
        >>> vincenty((0.0, 0.0), (0.0, 0.0))  # coincident points
        0.0
        >>> vincenty((0.0, 0.0), (0.0, 1.0))
        111.319491
        >>> vincenty((0.0, 0.0), (1.0, 0.0))
        110.574389
        >>> vincenty((0.0, 0.0), (0.5, 179.5))  # slow convergence
        19936.288579
        >>> vincenty((0.0, 0.0), (0.5, 179.7))  # failure to converge
        >>> boston = (42.3541165, -71.0693514)
        >>> newyork = (40.7791472, -73.9680804)
        >>> vincenty(boston, newyork)
        298.396057
        >>> vincenty(boston, newyork, miles=True)
        185.414657
        """

        #alpha1 in degrees
        alpha1=math.radians(alpha1)
        U1 = math.atan((1.0-f)*math.tan(math.radians(point1[0])))
        #print U1
        sigma1 = math.atan2((math.tan(U1)),(math.cos(alpha1)))
        sinAlpha=math.cos(U1)*math.sin(alpha1)
        cosSqAlpha=1.0-(sinAlpha**2)
        uSq = cosSqAlpha*(a**2-b**2)/(b**2)
        A = 1 + uSq/16384.0*(4096.0+uSq*(-768.0+uSq*(320.0-175*uSq)))
        B = uSq/1024*(256+uSq*(-128+uSq*(74-47*uSq)))

        sigma=s/b/A
        #print sigma
        for iteration in range(MAX_ITERATIONS):

            sigma2m = 2*sigma1+sigma
            deltasigma = B*math.sin(sigma)*(math.cos(sigma2m)+1.0/4*B*(math.cos(sigma)*(-1+2*(math.cos(sigma2m)**2))-1.0/6*B*math.cos(sigma2m)*(-3+4*(math.sin(sigma)**2))*(-3+4*(math.cos(sigma2m)**2))))
            sigmaprev = sigma
            sigma = s/b/A+deltasigma
            #print sigma
            if abs(sigma - sigmaprev) < CONVERGENCE_THRESHOLD:
                #print 'converge'
                break  # successful convergence
        else:
            print ('no converg')
            return None  # failure to converge


        num = math.sin(U1)*math.cos(sigma)+math.cos(U1)*math.sin(sigma)*math.cos(alpha1)
        den = (1.0-f)*math.sqrt(sinAlpha**2+(math.sin(U1)*math.sin(sigma)-math.cos(U1)*math.cos(sigma)*math.cos(alpha1))**2)
        #print num
        #print den
        lat2= math.atan2(num,den)

        num=math.sin(sigma)*math.sin(alpha1)
        den = math.cos(U1)*math.cos(sigma)-math.sin(U1)*math.sin(sigma)*math.cos(alpha1)
        Lambda = math.atan2(num,den)

        C = f/16.0*(cosSqAlpha*(4+f*(4.0-3.0*cosSqAlpha)))
        L = Lambda - (1.0-C)*f*sinAlpha*(sigma+C*math.sin(sigma)*(math.cos(sigma2m)+C*math.cos(sigma)*(-1+2.0*(math.cos(sigma2m)**2))))

        L2 = math.radians(point1[1])+L
        num = sinAlpha
        den = -1*math.sin(U1)*math.sin(sigma)+math.cos(U1)*math.cos(sigma)*math.cos(alpha1)
        #print num
        #print den
        alpha2 = math.atan2(num,den)
        if alpha2<0:
            alpha2+=math.pi*2
        #print alpha2
        # short-circuit coincident points
        return (math.degrees(lat2),math.degrees(L2)),math.degrees(alpha2)

    def reduce2mp(polys, verbose=False):
        ### generate a big multipolygon -> takes forever. Make small mps of 100 each then, combine.
        big_mps = []
        big_mp = polys[0]
        mod_count=0
        for ii in range(1,len(polys)):
            if ii%100==0:
                if verbose:
                    print ('mod count',ii)
                mod_count+=1
                big_mps.append(big_mp)
                big_mp=polys[ii]
            else:
                #print (mod_count,ii)
                big_mp=big_mp.union(polys[ii])
        big_mps.append(big_mp)

        if verbose:
            print ('n big mps',len(big_mps))

        ### now reduce list of big_mps
        big_mp=big_mps[0]
        for ii in range(1,len(big_mps)):
            if verbose:
                print ('big_mp: ',ii)
            big_mp = big_mp.union(big_mps[ii])
        return big_mp


    def extpolys2ext_w_holes(polys, return_dict = False):
        ### generate a big multipolygon -> takes forever. Make small mps of 100 each then, combine.
        big_mp = reduce2mp(polys)

        #big_mp = polys[0]
        #if len(polys)>0:
        #    for ii_p,pp in enumerate(polys[1:]):
        #        print (ii_p, len(polys))
        #        big_mp = big_mp.union(pp)
        #print (big_mp.type)

        if big_mp.type=='MultiPolygon':

            exterior_polys = list(big_mp)
            #print ('ext polys',len(exterior_polys))
            polys_dict = {k:[] for k in range(len(exterior_polys))}

            for ii_p,pp in enumerate(polys):
                for ii_e,ext in enumerate(exterior_polys):
                    #print (geometry.mapping(pp))
                    #print (geometry.mapping(ext))
                    #print ('amost eq',ii_p,ii_e, pp.almost_equals(ext))
                    if pp.within(ext) and not pp.almost_equals(ext):
                        polys_dict[ii_e].append(ii_p)
            #print (polys_dict)
                    #if pp.equals(ext):
                    #    print ('equal', ii_p, ii_e)
            polys_out = []

            for k in range(len(exterior_polys)):
                interior_polys = [polys[ii_p] for ii_p in polys_dict[k]]

                if len(interior_polys)>1:

                    int_mp = reduce2mp(interior_polys)
                    append_poly = exterior_polys[k]

                    if int_mp.type=='Polygon':
                        append_poly = append_poly.difference(int_mp)
                    elif int_mp.type=='MultiPolygon':
                        for int_p in list(int_mp):
                            append_poly = append_poly.difference(int_p)


                    polys_out.append(append_poly)

                elif len(interior_polys)>0:
                    polys_out.append(exterior_polys[k].difference(interior_polys[0]))

                else:
                    polys_out.append(exterior_polys[k])

            if return_dict:
                return polys_out, polys_dict
            else:
                return polys_out
            #print (polys_dict)
        else:
            if return_dict:
                return [big_mp], {k:[] for k in range(len([big_mp]))}
            else:
                return [big_mp]


    ### AREA

    from math import pi, sin
    WGS84_RADIUS = 6378137


    def rad(value):
        return value * pi / 180


    def ring__area(coordinates):
        """
        Calculate the approximate _area of the polygon were it projected onto
            the earth.  Note that this _area will be positive if ring is oriented
            clockwise, otherwise it will be negative.

        Reference:
            Robert. G. Chamberlain and William H. Duquette, "Some Algorithms for
            Polygons on a Sphere", JPL Publication 07-03, Jet Propulsion
            Laboratory, Pasadena, CA, June 2007 http://trs-new.jpl.nasa.gov/dspace/handle/2014/40409

        @Returns

        {float} The approximate signed geodesic _area of the polygon in square meters.
        """

        assert isinstance(coordinates, (list, tuple))

        _area = 0
        coordinates_length = len(coordinates)

        if coordinates_length > 2:
            for i in range(0, coordinates_length):
                if i == (coordinates_length - 2):
                    lower_index = coordinates_length - 2
                    middle_index = coordinates_length - 1
                    upper_index = 0
                elif i == (coordinates_length - 1):
                    lower_index = coordinates_length - 1
                    middle_index = 0
                    upper_index = 1
                else:
                    lower_index = i
                    middle_index = i + 1
                    upper_index = i + 2

                p1 = coordinates[lower_index]
                p2 = coordinates[middle_index]
                p3 = coordinates[upper_index]

                _area += (rad(p3[0]) - rad(p1[0])) * sin(rad(p2[1]))

            _area = _area * WGS84_RADIUS * WGS84_RADIUS / 2

        return _area


    def polygon__area(coordinates):

        assert isinstance(coordinates, (list, tuple))

        _area = 0
        if len(coordinates) > 0:
            _area += abs(ring__area(coordinates[0]))

            for i in range(1, len(coordinates)):
                _area -= abs(ring__area(coordinates[i]))

        return _area


    def area(geometry):

        if isinstance(geometry, str):
            geometry = json.loads(geometry)

        assert isinstance(geometry, dict)

        _area = 0

        if geometry['type'] == 'Polygon':
            return polygon__area(geometry['coordinates'])
        elif geometry['type'] == 'MultiPolygon':
            for i in range(0, len(geometry['coordinates'])):
                _area += polygon__area(geometry['coordinates'][i])

        elif geometry['type'] == 'GeometryCollection':
            for i in range(0, len(geometry['geometries'])):
                _area += area(geometry['geometries'][i])

        return _area

    #### </area
    def try_float(ff):
        try:
            float(ff)
            return True
        except:
            return False
    def parse_properties(ft):
        #print (ft['properties'])
        data = {}
        for pp,vv in ft['properties'].items():

            if pp.split(':')[0]=='sentinel-2':
                dd = (datetime.datetime.strptime(pp.split(':')[2][0:10], date_format) - baseline).days
                #print (dd,pp,vv)

                if try_float(vv):
                    if dd not in data.keys():
                        data[dd]={}
                    data[dd][pp.split(':')[-1]]=vv
                #else:
        return np.array([(k,data[k]['M'],data[k]['P']) for k in sorted(data.keys())])

    def prep_data_from_fc(fc, test=False):
        M = len(fc['features'])
        #print ('M',M)
        X_trn_tseries = -1.*np.ones((M,15,2))
        X_trn_aspect_ratio = np.zeros((M,1))
        X_trn_area = np.zeros((M,1))
        X_trn_ratio_perim_area = np.zeros((M,1))
        #print ('x_trn shape', X_trn_tseries.shape)


        for ii_f,ft in enumerate(fc['features']):
            data = parse_properties(ft)
            #print ('iif data.shape', ii_f, data.shape)
            if not data.shape[0]==0:
                #X_trn_tseries[ii_f,0:data.shape[0],:] = data[:,1:3]  #-> A
                X_trn_tseries[ii_f,(15-data.shape[0]):,:] = data[:,1:3]  #-> B
            #print (parse_properties(ft))
            #print ()
            if ft['geometry'] is not None:

                if ft['geometry']['type']=='Polygon':
                    ft_poly = geometry.Polygon(ft['geometry']['coordinates'][0])
                    #print ('ft_poly',ft_poly)
                    box_coords = np.array(ft_poly.minimum_rotated_rectangle.exterior.coords.xy)
                    l = V_inv((box_coords[0][0],box_coords[1][0]),(box_coords[0][1],box_coords[1][1]))[0]*1000
                    w = V_inv((box_coords[0][1],box_coords[1][1]),(box_coords[0][2],box_coords[1][2]))[0]*1000

                    X_trn_aspect_ratio[ii_f,0] = min(l,w)/max(l,w)
                    X_trn_area[ii_f,0] = area(ft['geometry'])
                    X_trn_ratio_perim_area[ii_f,0] = geometry.Polygon(ft['geometry']['coordinates'][0]).length*1000/area(ft['geometry'])
                else:
                    print (ft['geometry']['type'])


        X_trn_ratio_perim_area *= 100
        X_trn_area = (np.log(X_trn_area)/20.).clip(0.,1.)

        if not test:
            return X_trn_tseries,X_trn_aspect_ratio, X_trn_area, X_trn_ratio_perim_area
        else:
            Y_trn = np.zeros((M,1))
            for ii_f,ft in enumerate(fc['features']):

                Y_trn[ii_f,0] = ft['properties']['label']

            return X_trn_tseries,X_trn_aspect_ratio, X_trn_area, X_trn_ratio_perim_area, Y_trn

    def flatten_polys(polys):
        polys_flattened=[]
        for pp in polys:
            if pp.type=='MultiPolygon':
                polys_flattened+=list(pp)
            elif pp.type=='Polygon':
                polys_flattened+=[pp]
            else:
                print ('not poly',pp.type)
        return polys_flattened

    ### ACTUAL CODE STARTS HERE ###
    #
    #
    #
    #
    ###############################


    ### get the features

    tic = time.time()
    print ('tic',tic)

    tile_poly = geometry.shape(dltile['geometry'])
    tile_srs = dltile['properties']['proj4']
    dt = dltile['properties']['geotrans']

    vector_src = dl.vectors.FeatureCollection(src_vector_id)
    vector_dest = dl.vectors.FeatureCollection(dest_vector_id)

    tile_fts = [geojson.Feature(geometry=ft.geometry,properties=ft.properties) for ft in vector_src.filter(tile_poly).features()]
    print ('get fts toc', time.time()-tic)

    if len(tile_fts)==0:
        return json.dumps(geojson.FeatureCollection(None,properties={'FLAG':'SUCCESS','uploaded_fts':0}))


    ### reduce to remove holes
    ft_polys = [geometry.shape(ft['geometry']) for ft in tile_fts]
    new_polys, new_polys_dict = extpolys2ext_w_holes(ft_polys, return_dict = True)
    print ('remove holes toc', time.time()-tic)

    tile_fts = [geojson.Feature(geometry=new_polys[ii], properties=tile_fts[ii]['properties']) for ii in new_polys_dict.keys()]


    print ('holes reduced:', len(ft_polys), len(new_polys))


    ### get the model
    json_str = storage_client.get('model_json_ftfilter_B', storage_type='data')

    model = keras.models.model_from_json(json_str)

    model_weight_shapes = dict(json.loads(storage_client.get('model_weight_shapes_ftfilter_B',storage_type='data')))

    rebuild_weights = []

    for ii in range(len(model_weight_shapes.keys())):
        w = np.frombuffer(storage_client.get('model_weights_ftfilter_B_'+str(ii), storage_type='data'), dtype='float32').reshape(model_weight_shapes[str(ii)])

        rebuild_weights.append(w)

    model.set_weights(rebuild_weights)

    print ('get model toc',time.time()-tic)

    ### run the features through the model

    X_test_tseries,X_test_aspect_ratio, X_test_area, X_test_ratio_perim_area = prep_data_from_fc(geojson.FeatureCollection(tile_fts), test=False)
    y_hat = model.predict([X_test_tseries,X_test_aspect_ratio, X_test_area, X_test_ratio_perim_area])

    for ii_f, ft in enumerate(tile_fts):
        ft['properties']['prediction']=float(y_hat[ii_f])

    print ('get prediction toc', time.time()-tic)

    ### re-create the image
    if push_rast:
        WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        annotations = np.zeros((dltile['properties']['tilesize'], dltile['properties']['tilesize'])) #np.ones((arr.shape[0], arr.shape[1]))*128
        im = Image.fromarray(annotations, mode='L')
        draw = ImageDraw.Draw(im)

        wgs_proj = pyproj.Proj(WGS84)
        utm_proj = pyproj.Proj(tile_srs)

        for ft in tile_fts:
            if ft['geometry'] is not None:
                ll_poly = geometry.shape(ft['geometry'])

                projection_func = partial(pyproj.transform, wgs_proj, utm_proj)
                utm_polys = flatten_polys([transform(projection_func, ll_poly)])

                for utm_poly in utm_polys:

                    pix_poly = geometry.Polygon([lonlat2pixXY(c,dt) for c in list(utm_poly.exterior.coords)])
                    xs,ys = pix_poly.exterior.xy
                    #print (int(np.round(ft['properties']['prediction']*255,0)))
                    draw.polygon(list(zip(xs,ys)),fill=int(np.round(ft['properties']['prediction']*255,0)))

                    print ('interiors:', len(utm_poly.interiors))
                    if len(utm_poly.interiors)>0:
                        for int_ring in utm_poly.interiors:
                            pix_poly = geometry.Polygon([lonlat2pixXY(c,dt) for c in list(int_ring.coords)])
                            xs,ys = pix_poly.exterior.xy
                            #print (int(np.round(ft['properties']['prediction']*255,0)))
                            draw.polygon(list(zip(xs,ys)),fill=0)

        annotations = np.array(im).astype('uint8')

        print ('draw annotations toc',time.time()-tic)

        meta = dltile['properties']
        print ('meta',meta)
        print ('annoations shape',annotations.shape)

        image_id = ':'.join([str(datetime.date.today()),'primary_vector_reraster', dltile['properties']['key'].replace(':', '_')])

        time.sleep(5)

        print ('uploading image...')

        cat_return = catalog_client.upload_ndarray(annotations, dest_product_id, image_id, proj4 =tile_srs, geotrans=dt, overviews=[5,10], overview_resampler='average')
        #raster_meta=meta,

        print ('catalog return')
        print (cat_return)
        print ('pushing catalog toc', time.time()-tic)


    ### re-upload the features

    new_features = []
    for ii_f, ft in enumerate(tile_fts):
        if ft['geometry'] is not None:
            print (ii_f, np.round(ft['properties']['prediction']*255,0),geometry.shape(ft['geometry']).is_valid)
            if np.round(ft['properties']['prediction']*255,0)>=1.:
                new_features.append(
                    dl.vectors.Feature(
                        geometry=geometry.shape(ft['geometry']),
                        properties=ft['properties']
                    ))

    print ('len new features', len(new_features))


    if len(new_features)>0:
        #print ('new fts')
        #print (new_features)
        index= 0
        step=25
        try:
            while (index*step)<len(new_features):
                vector_dest.add(new_features[index*step:min((index+1)*step,len(new_features))])
                index+=1
                print ('upload index', index)
        except Exception as e:
            print ('error:',e)
            broken_fts = []
            uploaded_fts = 0
            for ft in new_features:
                try:
                    vector_dest.add([ft])
                    uploaded_fts+=1
                except:
                    broken_fts.append(geojson.Feature(geometry=ft.geometry,properties=ft.properties))

            out_fc = geojson.FeatureCollection(broken_fts, properties={'FLAG':'FAILURE','uploaded_fts':uploaded_fts})
            return json.dumps(out_fc)



    out_fc = geojson.FeatureCollection(None,properties={'FLAG':'SUCCESS','uploaded_fts':len(new_features)})
    return json.dumps(out_fc)

def SPOTVectoriser(dltile, src_product_id, band_names, scales, dest_fc_id,shp_str):
    import numpy as np
    import json, geojson, time, datetime
    from io import StringIO
    from shapely import geometry
    from shapely.ops import transform,linemerge, unary_union, polygonize
    from shapely.affinity import affine_transform
    from functools import partial
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    from skimage import measure
    from pyproj import Proj
    import pyproj
    from scipy import ndimage, spatial
    from scipy.signal import fftconvolve
    from PIL import Image, ImageDraw

    import descarteslabs as dl

    #load all the clients
    catalog_client = dl.Catalog()
    raster_client = dl.Raster()
    metadata_client = dl.Metadata()
    storage_client = dl.Storage()


    WATERSHED_HIGH_1=0.89
    WATERSHED_LOW_1 = 0.83
    GAUSS_KERNEL_1 = 1

    WATERSHED_THRESHOLD = 0.59
    WATERSHED_KERNEL = 5
    NDVI_FILTER = 0.718*255
    SIZE_FILTER_KW = 85

    WATERSHED_HIGH_2=0.84
    WATERSHED_LOW_2 = 0.18
    GAUSS_KERNEL_2 = 301

    ### AREA

    from math import pi, sin
    WGS84_RADIUS = 6378137


    def rad(value):
        return value * pi / 180


    def ring__area(coordinates):
        """
        Calculate the approximate _area of the polygon were it projected onto
            the earth.  Note that this _area will be positive if ring is oriented
            clockwise, otherwise it will be negative.

        Reference:
            Robert. G. Chamberlain and William H. Duquette, "Some Algorithms for
            Polygons on a Sphere", JPL Publication 07-03, Jet Propulsion
            Laboratory, Pasadena, CA, June 2007 http://trs-new.jpl.nasa.gov/dspace/handle/2014/40409

        @Returns

        {float} The approximate signed geodesic _area of the polygon in square meters.
        """

        assert isinstance(coordinates, (list, tuple))

        _area = 0
        coordinates_length = len(coordinates)

        if coordinates_length > 2:
            for i in range(0, coordinates_length):
                if i == (coordinates_length - 2):
                    lower_index = coordinates_length - 2
                    middle_index = coordinates_length - 1
                    upper_index = 0
                elif i == (coordinates_length - 1):
                    lower_index = coordinates_length - 1
                    middle_index = 0
                    upper_index = 1
                else:
                    lower_index = i
                    middle_index = i + 1
                    upper_index = i + 2

                p1 = coordinates[lower_index]
                p2 = coordinates[middle_index]
                p3 = coordinates[upper_index]

                _area += (rad(p3[0]) - rad(p1[0])) * sin(rad(p2[1]))

            _area = _area * WGS84_RADIUS * WGS84_RADIUS / 2

        return _area


    def polygon__area(coordinates):

        assert isinstance(coordinates, (list, tuple))

        _area = 0
        if len(coordinates) > 0:
            _area += abs(ring__area(coordinates[0]))

            for i in range(1, len(coordinates)):
                _area -= abs(ring__area(coordinates[i]))

        return _area


    def area(geometry):

        if isinstance(geometry, str):
            geometry = json.loads(geometry)

        assert isinstance(geometry, dict)

        _area = 0

        if geometry['type'] == 'Polygon':
            return polygon__area(geometry['coordinates'])
        elif geometry['type'] == 'MultiPolygon':
            for i in range(0, len(geometry['coordinates'])):
                _area += polygon__area(geometry['coordinates'][i])

        elif geometry['type'] == 'GeometryCollection':
            for i in range(0, len(geometry['geometries'])):
                _area += area(geometry['geometries'][i])

        return _area

    #### </area

    def reduce2mp(polys, verbose=False):
        ### generate a big multipolygon -> takes forever. Make small mps of 100 each then, combine.
        big_mps = []
        big_mp = polys[0]
        mod_count=0
        for ii in range(1,len(polys)):
            if ii%100==0:
                if verbose:
                    print ('mod count',ii)
                mod_count+=1
                big_mps.append(big_mp)
                big_mp=polys[ii]
            else:
                #print (mod_count,ii)
                big_mp=big_mp.union(polys[ii])
        big_mps.append(big_mp)

        if verbose:
            print ('n big mps',len(big_mps))

        ### now reduce list of big_mps
        big_mp=big_mps[0]
        for ii in range(1,len(big_mps)):
            #print ('big_mp: ',ii)
            big_mp = big_mp.union(big_mps[ii])
        return big_mp

    def watershed_from_local_maxi(inp_arr,trim_low, trim_high,local_maxi,ws_threshold,gauss_kernel):
        #inp_arr is a normalised array 0.:1.
        #local_maxi is a boolean mask with shape inp_arr with peak=True
        mask_arr = np.copy(inp_arr)
        mask_arr = gaussian_blur(mask_arr, gauss_kernel)
        #mask_arr = np.copy(inp_arr)

        mask_arr[mask_arr>trim_high] =1.
        mask_arr[mask_arr<trim_low] = 0.
        distance = ndimage.distance_transform_edt(mask_arr)

        markers = ndimage.label(local_maxi)[0]

        labels = watershed(-distance,
                           markers, mask=mask_arr>((trim_high-trim_low)*ws_threshold+trim_low))

        return labels

    def watershed_labels2(inp_arr,trim_low, trim_high,kernel):
        #inp_arr is a normalised array 0.:1.
        inp_arr[inp_arr>trim_high] =1.
        inp_arr[inp_arr<trim_low] = 0.
        distance = ndimage.distance_transform_edt(inp_arr)

        print ('ws sums',np.sum(distance>0), np.sum(inp_arr>0))

        try:
            local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((kernel, kernel)),labels=inp_arr)
        except Exception as e:
            print ('error in local maxi: ',e)
            local_maxi = None

        if local_maxi is None:
            return None
        else:

            markers = ndimage.label(local_maxi)[0]
            labels = watershed(-distance,
                           markers, mask=inp_arr)

            return labels

    def gaussian_blur(in_array, size):
        # expand in_array to fit edge of kernel
        padded_array = np.pad(in_array, size, 'symmetric')
        # build kernel
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
        g = (g / g.sum()).astype(in_array.dtype)
        # do the Gaussian blur
        return fftconvolve(padded_array, g, mode='valid')

    def vectorise_2(arr, tile_size, disp=2):

        polys = []

        contours = measure.find_contours(np.pad(arr,disp,'constant', constant_values=(0)), 0.8)
        for c in contours:
            #print ('c shape',c.shape)
            #print ('assrt', c[:,:]==c[1])

            c = (c-disp).clip(0.,float(tile_size))
            poly = geometry.Polygon(c)
            poly = poly.simplify(0.1)
            polys.append(poly)
        return polys

    def pixXY2lonlat(pt,dt):
        X = pt[0]
        Y = pt[1]
        lon = X*dt[1]+dt[0]+Y*dt[2]
        #print Y
        lat = Y*(dt[5]-dt[2]*dt[4]/dt[1])+dt[3]+(lon-dt[0])*dt[4]/dt[1]
        #print X
        return [lon,lat]

    def pixvec2rast(polys, tilesize):

        blank = np.zeros((tilesize, tilesize))
        im = Image.fromarray(blank, mode='L')
        draw = ImageDraw.Draw(im)

        for p in polys:
            xs, ys = p.exterior.xy
            draw.polygon(list(zip(xs,ys)), fill=255)

        return np.array(im)

    def lonlat2pixXY(pt,dt):
        lon = pt[0]
        lat = pt[1]
        Y = (lat-dt[3]-dt[4]/dt[1]*(lon-dt[0]))/(dt[5]-dt[2]*dt[4]/dt[1])
        #print Y
        X = (lon-dt[0]-Y*dt[2])/dt[1]
        #print (lon,dt[0],X)
        #print X
        return [int(X),int(Y)]

    def flatten_polys(polys):
        polys_flattened=[]
        for pp in polys:
            if pp.type=='MultiPolygon':
                polys_flattened+=list(pp)
            elif pp.type=='Polygon':
                polys_flattened+=[pp]
            elif pp.type=='GeometryCollection' and not pp.is_empty:
                for subpp in list(pp):
                    if subpp.type=='MultiPolygon':
                        polys_flattened+=list(subpp)
                    elif subpp.type=='Polygon':
                        polys_flattened+=[subpp]
            else:
                print ('not poly',pp.type)
        return polys_flattened

    def extpolys2ext_w_holes(polys):
        ### generate a big multipolygon -> takes forever. Make small mps of 100 each then, combine.
        big_mp = reduce2mp(polys)

        #big_mp = polys[0]
        #if len(polys)>0:
        #    for ii_p,pp in enumerate(polys[1:]):
        #        print (ii_p, len(polys))
        #        big_mp = big_mp.union(pp)
        #print (big_mp.type)

        if big_mp.type=='MultiPolygon':

            exterior_polys = list(big_mp)
            print ('ext polys',len(exterior_polys))
            polys_dict = {k:[] for k in range(len(exterior_polys))}

            for ii_p,pp in enumerate(polys):
                for ii_e,ext in enumerate(exterior_polys):
                    #print (geometry.mapping(pp))
                    #print (geometry.mapping(ext))
                    #print ('amost eq',ii_p,ii_e, pp.almost_equals(ext))
                    if pp.within(ext) and not pp.almost_equals(ext):
                        polys_dict[ii_e].append(ii_p)
            print (polys_dict)
                    #if pp.equals(ext):
                    #    print ('equal', ii_p, ii_e)
            polys_out = []

            for k in range(len(exterior_polys)):
                interior_polys = [polys[ii_p] for ii_p in polys_dict[k]]

                if len(interior_polys)>1:

                    int_mp = reduce2mp(interior_polys)
                    append_poly = exterior_polys[k]

                    if int_mp.type=='Polygon':
                        append_poly = append_poly.difference(int_mp)
                    elif int_mp.type=='MultiPolygon':
                        for int_p in list(int_mp):
                            append_poly = append_poly.difference(int_p)


                    polys_out.append(append_poly)

                elif len(interior_polys)>0:
                    polys_out.append(exterior_polys[k].difference(interior_polys[0]))

                else:
                    polys_out.append(exterior_polys[k])

            return polys_out
            #print (polys_dict)
        else:
            return [big_mp]


    ###### PREP TILE #######
    #################3######

    #fetch the scene metadata
    print ('dltile key:',dltile['properties']['key'])

    scenes = metadata_client.search(src_product_id, geom=raster_client.dltile(dltile['properties']['key']), start_datetime='2016-01-01',  end_datetime='2019-12-29', limit=15)['features']

    #get the most recent one
    scenes = sorted(scenes, key=lambda k: k.properties.acquired, reverse=True) #do most recent scene first

    #print (scenes)
    print ('n scenes: ', len(scenes))

    if len(scenes)<1:
        outp = {
        'message':'no scenes',
        'storage_bit':0
        }
        return json.dumps(outp)

    tile_dimension = dltile['properties']['tilesize']+2*dltile['properties']['pad']

    ### make the tile array
    tile_arr = np.zeros((len(scenes),tile_dimension,tile_dimension))

    scene_ind = 0
    fill_frac = 0.

    for ii_s, s in enumerate(scenes):
        new_arr, meta = raster_client.ndarray(
            s['id'],
            bands=band_names,
            scales=scales,
            ot='Byte',
            dltile=dltile['properties']['key']
        )

        tile_arr[ii_s, :,:] = new_arr

    tile_arr = np.amax(tile_arr, axis=0)/255.

    tile_poly = geometry.shape(dltile['geometry'])

    ### Fetch NDVI array and filter

    #manage summer/winter
    if tile_poly.centroid.y>=0.:
        sdate = '2018-06-01'
        edate = '2018-08-31'
    else:
        sdate = '2018-12-01'
        edate = '2019-02-28'

    print ('seasons',tile_poly.centroid.y, sdate, edate)

    scenes_s2 = metadata_client.search('sentinel-2:L1C', geom=raster_client.dltile(dltile['properties']['key']), start_datetime=sdate,  end_datetime=edate,cloud_fraction=0.2, limit=15)['features']

    print ('S2 scenes', len(scenes_s2))

    if len(scenes_s2)>0:
        scenes_s2=sorted(scenes_s2, key=lambda k: k.properties.cloud_fraction, reverse=False)

        ndvi_arr, dummy = raster_client.ndarray(
                scenes_s2[0]['id'],
                bands=['derived:ndvi'],
                scales=[[0,65535]],
                ot='Float32',
                dltile=dltile['properties']['key']
            )

        print ('ndvi shape, max, min',ndvi_arr.shape, np.max(ndvi_arr), np.min(ndvi_arr))

        tile_arr[ndvi_arr>NDVI_FILTER]=0

    ### Remove coastlines
    #get the big shp poly
    dt = dltile['properties']['geotrans']
    dt_shapely = [dt[1],dt[2],dt[4],dt[5],dt[0],dt[3]]

    print ('dt',dt)

    coast_shp = geometry.shape(json.loads(storage_client.get('country_shape_'+shp_str, storage_type='data')))
    #coast_shp = geometry.shape(places_client.shape(slug_str)['geometry'])
    print('got shp')
    coast_mask = np.zeros(tile_arr.shape)
    tile_srs = dltile['properties']['proj4']

    WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

    im = Image.fromarray(coast_mask, mode='L')
    draw = ImageDraw.Draw(im)

    wgs_proj = pyproj.Proj(WGS84)
    utm_proj = pyproj.Proj(tile_srs)

    print ('got projs')


    tile_int = tile_poly.intersection(coast_shp) #big_shp_mp

    if tile_int.type=='MultiPolygon':
        tile_int_pp = list(tile_int)
    elif tile_int.type=='Polygon':
        tile_int_pp = [tile_int]
    else:
        outp = {
        'message':'no tile intersection',
        'storage_bit':0
        }
        return json.dumps(outp)

    projection_func = partial(pyproj.transform, wgs_proj, utm_proj)
    print ('got projections')

    for pp in tile_int_pp:
        utm_poly = transform(projection_func, pp)

        ## ext poly
        pix_poly = geometry.Polygon([lonlat2pixXY(c,dt) for c in list(utm_poly.exterior.coords)])
        xs,ys = pix_poly.exterior.xy
        #print (int(np.round(ft['properties']['prediction']*255,0)))
        draw.polygon(list(zip(xs,ys)),fill=255)

        ## int poly
        if pp.interiors.__len__() >0:
            int_polys = [geometry.Polygon(pp.interiors._get_ring(ii).coords) for ii in range(pp.interiors.__len__()) ]
            for int_pp in int_polys:
                utm_poly = transform(projection_func, int_pp)

                pix_poly = geometry.Polygon([lonlat2pixXY(c,dt) for c in list(utm_poly.exterior.coords)])
                xs,ys = pix_poly.exterior.xy
                #print (int(np.round(ft['properties']['prediction']*255,0)))
                draw.polygon(list(zip(xs,ys)),fill=0)

    print ('clipped coastline')


    ### mask water

    pekel_prod = catalog_client.get_product('a8047a8ea57370d6356e594061a457fbc7d7a082:water:occurrence:v0')

    scenes_pekel = metadata_client.search(pekel_prod['data']['id'],
                                          geom=raster_client.dltile(dltile['properties']['key']),
                                          start_datetime='2015-01-01',
                                          end_datetime='2019-12-31')['features']
    print ('n scenes peke', len(scenes_pekel))

    if len(scenes_pekel)>0:

        pekel_arr = np.zeros((len(scenes_pekel),tile_dimension,tile_dimension))

        for ii_s, s in enumerate(scenes_pekel):
            new_arr, dummy = raster_client.ndarray(
                s['id'],
                bands=['occurrence'],
                scales=[[0,255]],
                ot='Byte',
                dltile=dltile['properties']['key']
            )

            pekel_arr[ii_s, :,:] = new_arr

        pekel_arr = np.amax(pekel_arr, axis=0)

        tile_arr[pekel_arr>=10]=0

    ##### VECTORISE 1 #####
    #######################


    ### Vectorise Prediction
    gauss_arr = gaussian_blur(tile_arr,GAUSS_KERNEL_1)
    print (gauss_arr.shape)

    labels = watershed_labels2(gauss_arr,WATERSHED_LOW_1, WATERSHED_HIGH_1,WATERSHED_KERNEL)

    if labels is None:
        print ('no vector1 labels')
        outp = {
        'message':'no vector1 labsl',
        'storage_bit':0
        }
        return json.dumps(outp)

    unique_labels = np.unique(labels)

    unique_labels = np.delete(unique_labels, np.argwhere(unique_labels==0))

    polys = []
    for ul in unique_labels:

        v2 = vectorise_2(labels==ul, tile_dimension,5)
        for v in v2:
            polys.append(v)


    #for pp in polys:
    #    #xs,ys = pp.exterior.xy
    #    #axs[1].plot(xs,ys, color='b')

    print ('tile goemetry')
    print (dltile['geometry'])

    res = dltile['properties']['resolution']
    dt = dltile['properties']['geotrans']

    print (dt)

    if len(polys)==0:
        print ('no vector1 polys')
        outp = {
        'message':'vector1 polys',
        'storage_bit':0
        }
        return json.dumps(outp)

    ####### VECTORISE 2 ######
    ##########################

    local_peaks = []
    rep_pts = []

    for pp in polys:
        rep_pt = pp.representative_point()
        local_peaks.append([rep_pt.x, rep_pt.y])
        #axs[1].scatter(rep_pt.x, rep_pt.y, color='g',s=50)
        #axs[2].scatter(rep_pt.x, rep_pt.y, color='g',s=50)

    local_peaks_arr = np.zeros(tile_arr.shape)
    for pt in local_peaks:
        #print (pt)
        local_peaks_arr[int(min(pt[0],tile_dimension-1)),int(min(pt[1],tile_dimension-1))]=1



    ### Vectorise Prediction from seed locations

    filter_polys = []
    filter_polys_ll = []


    labels = watershed_from_local_maxi(tile_arr,WATERSHED_LOW_2, WATERSHED_HIGH_2,local_peaks_arr>0,WATERSHED_THRESHOLD, GAUSS_KERNEL_2)

    unique_labels = np.unique(labels)
    unique_labels = np.delete(unique_labels, np.argwhere(unique_labels==0))
    for ul in unique_labels:
        filter_polys += vectorise_2((labels==ul).T, tile_dimension,5)

    print ('len orig vector polys',len(filter_polys))

    ext_polys_mp = reduce2mp([pp.buffer(5) for pp in filter_polys]).buffer(-5)
    if ext_polys_mp.type=='MultiPolygon':
        ext_polys = list(ext_polys_mp)
    elif ext_polys_mp.type=='Polygon':
        ext_polys = [ext_polys_mp]
    else:
        print (ext_polys_mp.type)
        print (ext_polys_mp)
        raise Exception('ext polys geom type')

    print ('ext_polys',len(ext_polys))
    #print (ext_polys[0])


    ext_polys_utm = [affine_transform(pp,dt_shapely) for pp in ext_polys]

    print ('ext_utm',len(ext_polys_utm))
    #print (ext_polys_utm[0])


    WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    tile_srs = dltile['properties']['proj4']

    utm_proj = pyproj.Proj(tile_srs)
    wgs_proj = pyproj.Proj(WGS84)

    projection_func = partial(pyproj.transform, utm_proj,wgs_proj)
    ext_polys_ll = []
    for ext_poly_utm in ext_polys_utm:
        ext_polys_ll += flatten_polys([transform(projection_func, ext_poly_utm)])
    print ('ext_ll',len(ext_polys_ll))




    features = []
    fc_dest = dl.vectors.FeatureCollection(dest_fc_id)
    failed_fts = []
    for ii_p, pp in enumerate(ext_polys_ll):

        if area(geometry.mapping(pp))>(SIZE_FILTER_KW*1000/360*1.6):

            properties = {'primary_id':dltile['properties']['key']+'_'+str(ii_p)}
            print (properties, pp.is_valid, 'holes:',len(pp.interiors))

            if pp.is_valid:

                properties['dltile']=dltile['properties']['key']

                features.append(dl.vectors.Feature(
                        geometry=pp.simplify(1e-5),
                        properties=properties
                    ))

            else:
                print ('invalid geom')
        else:
            print ('area too small:',area(geometry.mapping(pp)))


    if len(features)>0:

        try:

            index= 0
            step=20
            while (index*step)<len(features):
                print ('upload index',index)
                fc_dest.add(features[index*step:min((index+1)*step,len(features))])
                index+=1
        except Exception as e:
            print ('error!')
            print (e)
            print ('creating upload task')
            f = StringIO('\n'.join([json.dumps(f.geojson) for f in features[index*step:]]))
            fc_dest.upload(f)


    output = {}
    output['n_scenes'] = len(scenes)
    output['n_features'] = len(features)



    return json.dumps(output) #scene_check_ids

DL_CLOUD_FUNCTIONS = {
    'S2Infer1':S2Infer1,
    'S2Infer2':S2Infer2,
    'S2RNN1':S2RNN1,
    'SPOTVectoriser':SPOTVectoriser,
}
