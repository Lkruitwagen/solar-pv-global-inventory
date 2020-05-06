import json, geojson, os, sys
import numpy as np
from scipy import stats
from shapely import geometry
import datetime as dt

import descarteslabs as dl
import descarteslabs.workflows as wf


def _mp_worker(fc_path,out_path,do_region):
    
    raster_client = dl.Raster()
    metadata_client = dl.Metadata()


    corine_countries = ['AL', 'AT', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IS', 'IE', 'IT', 'XK', 'LV', 'LI', 'LT', 'LU', 'MK', 'MT', 'ME', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'TR', 'GB']

    fts = json.load(open(fc_path,'r'))['features']

    if do_region=='US':

        run_fts = [ft for ft in fts if ft['properties']['iso-3166-1'] in ['US']]
    elif do_region=='CN':
        run_fts = [ft for ft in fts if ft['properties']['iso-3166-1'] in ['CN']]
    elif do_region=='CORINE':
        run_fts = [ft for ft in fts if ft['properties']['iso-3166-1'] in corine_countries]
    elif do_region=='REMAINDER':
        run_fts = [ft for ft in fts if ft['properties']['iso-3166-1'] not in corine_countries+['US','CN']]

    print (do_region, len(run_fts))


    image_collections = {}


    if do_region=='US':

        key = 'land_cover_CDL_'

        start_dates = [str(yr)+'-12-30' for yr in range(2005,2018)]

        for date in start_dates:
            sdate = dt.datetime.strptime(date,'%Y-%m-%d')
            edate = sdate + dt.timedelta(days=3)

            image_collections[edate.isoformat()[0:4]] = wf.ImageCollection.from_id("usda:cdl:v1", 
                                                        start_datetime=sdate.isoformat()[0:10], 
                                                        end_datetime=edate.isoformat()[0:10])

    elif do_region=='CORINE':

        key = 'land_cover_CORINE_'

        start_dates = ['2006-12-30', '2012-12-30', '2018-12-30']

        for date in start_dates:
            sdate = dt.datetime.strptime(date,'%Y-%m-%d')
            edate = sdate + dt.timedelta(days=3)

            image_collections[sdate.isoformat()[0:4]] = wf.ImageCollection.from_id("oxford-university:corine-land-cover", 
                                                        start_datetime=sdate.isoformat()[0:10], 
                                                        end_datetime=edate.isoformat()[0:10])


    else:

        key = 'land_cover_MODIS_'

        start_dates = start_dates = [str(yr)+'-12-30' for yr in range(2005,2013)]

        for date in start_dates:
            sdate = dt.datetime.strptime(date,'%Y-%m-%d')
            edate = sdate + dt.timedelta(days=3)

            image_collections[edate.isoformat()[0:4]] = {'product':"modis:mcd12q1:051",
                                                         'bands':['Land_Cover_Type_1'],
                                                        'sdate':sdate.isoformat()[0:10], 
                                                        'edate':edate.isoformat()[0:10]}

        start_dates = start_dates = [str(yr)+'-12-30' for yr in range(2013,2018)]

        for date in start_dates:
            sdate = dt.datetime.strptime(date,'%Y-%m-%d')
            edate = sdate + dt.timedelta(days=3)

            image_collections[edate.isoformat()[0:4]] = {'product':"oxford-university:modis-land-cover", 
                                                         'bands':['IGBP_class'],
                                                        'sdate':sdate.isoformat()[0:10], 
                                                        'edate':edate.isoformat()[0:10]}

    print (key, image_collections.keys())
            
    for ii_f, ft in enumerate(run_fts):
        
        if key=='land_cover_MODIS_':
            pt = geometry.shape(ft['geometry']).representative_point()
            tile = raster_client.dltile_from_latlon(pt.y, pt.x, 500, 1, 0)
        
        if ii_f % 100==0:
            print (do_region, ii_f)
            
        if do_region in ['US','CN','REMAINDER']:
            yr_range = range(2006, 2019)
        elif do_region=='CORINE':
            yr_range = [2006, 2012, 2018]

        for yr in yr_range:
            
            try:
                
                if key=='land_cover_MODIS_':
                    
                    scenes = metadata_client.search(image_collections[str(yr)]['product'], 
                                                    geom=tile['geometry'], 
                                                    start_datetime=image_collections[str(yr)]['sdate'],  
                                                    end_datetime=image_collections[str(yr)]['edate'])['features']
                    
                    arr, meta = raster_client.ndarray(scenes[0].id, bands=image_collections[str(yr)]['bands'], scales=[[0,255]], ot='Byte', dltile=tile['properties']['key'])
                    
                    class_mode = int(stats.mode(arr.flatten()).mode[0])
                    
                    if (yr<2014) and (class_mode==0):
                        class_mode=17
                        
                else:
                    class_map = image_collections[str(yr)].compute(geoctx=wf.GeoContext(geometry=ft['geometry'],
                                                                resolution=10.,
                                                                crs='EPSG:3857'))
                    class_mode = int(stats.mode(class_map.ndarray.data[class_map.ndarray.mask==0]).mode[0])

                #print (yr,'class mode',class_mode)
                
            except Exception as e:
                print ('Error!', e)
                class_mode = 'null'

            print (yr, class_mode)

            ft['properties'][key+str(yr)] = class_mode


    json.dump(geojson.FeatureCollection(run_fts), open(os.path.join(out_path,do_region+'_FC.geojson'), 'w'))



if __name__=='__main__':
    fc_path = os.path.join(os.getcwd(), 'data', 'ABCD_simplified.geojson')
    out_path = os.path.join(os.getcwd(),'data')
    do_region = sys.argv[1]
    if do_region not in ['CORINE','REMAINDER','US','CN']:
        print ('not valid region!')
    else:
        _mp_worker(fc_path,out_path,do_region)