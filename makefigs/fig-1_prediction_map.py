### Figure 2 - Results Timeseries
import os, json, logging
from shapely import geometry
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure
from area import area
import descarteslabs as dl
from shapely.strtree import STRtree

from shapely.ops import transform
from functools import partial
import pyproj

catalog_client = dl.Catalog()
raster_client = dl.Raster()
metadata_client = dl.Metadata()
storage_client = dl.Storage()

class Fig1_generator:
    """
    A class to generate Figure 2. Options are:
    geography: one of ['WORLD',
                        OR an iso for a country (e.g. 'CN','DE','US'),
                        OR an iso-3166-2 for a state or province (e.g. 'CN-65','US-CA','CA-ON')]
    max_date: the maximum date to plot until, in 10char YYYY-mm-dd iso-format, e.g. '2018-03-06'
    window (optional): a latlon window to plot. Will override automatic window generation. [minlon, minlat, maxlon, maxlat]
    """



    def __init__(self):
        logging.info(f'Initialising...')

        self.data_paths = {
            'pts':os.path.join('data','fig_1_pts.csv'),
            'features':os.path.join('data','ABCD_simplified.geojson'),
            }

        self.colors = {
            'border':'#838487',
            'focus_fill':'#E5E5E5',#'#F0EBDD','#E5E5E5'
            'nonfocus_fill':'#8f8f8f',##DACBA6',#E5E5E5
            'background':'#e8f1ff',

            }

        self.products = {
            'S2':'sentinel-2:L1C',
            'SPOT':'airbus:oneatlas:spot:v2',
            'S2_pred_map':'ba611607613832ad7bb8fa9dc2bafb71f693bd6a:solar_pv:s2:v3_20190306:primary',
            'SPOT_pred_map':'8514dad6c277e007cedb6fb8e829a23c8975fca4:solar_pv:airbus:spot:v5_0111',
        }
        self.bands = {
        'SPOT':['red','green','blue','alpha'],
        'S2':['red','green','blue','alpha'],
        'SPOT_pred_map':['probability'],
        'S2_pred_map':['probability']
        }
        self.resolution = {
        'SPOT':1.5,
        'S2':10,
        'SPOT_pred_map':1.5,
        'S2_pred_map':10
        }
        self.tilesize = {
        'SPOT':1000,
        'S2':150,
        'SPOT_pred_map':1000,
        'S2_pred_map':150
        }
        self.sdate = {
        'SPOT':'2015-07-01',
        'S2':'2018-07-01',
        'SPOT_pred_map':'2015-07-01',
        'S2_pred_map':'2018-07-01'
        }

        self.edate = {
        'SPOT':'2018-12-31',
        'S2':'2019-12-31',
        'SPOT_pred_map':'2018-12-31',
        'S2_pred_map':'2019-12-31',
        }
        self.scales = {
        'SPOT':[[0,255]]*4,
        'S2':[[0,10000]]*4,
        'SPOT_pred_map':[[0,255]],
        'S2_pred_map':[[0,255]]
        }


        logging.info(f'Loading feature collection...')
        self.features = json.load(open(self.data_paths['features'],'r'))['features']
        self.ft_polys = [geometry.shape(ft['geometry']) for ft in self.features]
        self.poly_tree = STRtree(self.ft_polys)

        self.points = pd.read_csv(self.data_paths['pts']).to_records('dict')
        self.ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        #shortlist: [0,1,3,4,7,8,9,10,14,15]
        self.select_points = [0,1,3,7,9]#range(len(self.points))


    def pixXY2lonlat(self,pt,dt):
        X = pt[0]
        Y = pt[1]
        lon = X*dt[1]+dt[0]+Y*dt[2]
        #print Y
        lat = Y*(dt[5]-dt[2]*dt[4]/dt[1])+dt[3]+(lon-dt[0])*dt[4]/dt[1]
        #print X
        return [lon,lat]

    def lonlat2pixXY(self,pt,dt):
        lon = pt[0]
        lat = pt[1]
        Y = (lat-dt[3]-dt[4]/dt[1]*(lon-dt[0]))/(dt[5]-dt[2]*dt[4]/dt[1])
        #print Y
        X = (lon-dt[0]-Y*dt[2])/dt[1]
        #print (lon,dt[0],X)
        #print X
        return [int(X),int(Y)]

    def get_pix_polys(self,tile):
        tile_poly = geometry.shape(tile['geometry'])

        int_polys = self.poly_tree.query(tile_poly)
        int_polys = [pp for pp in int_polys if pp.intersects(tile_poly)]
        print (tile['geometry'])
        print ('int polys',len(int_polys))



        ll2utm = pyproj.Proj(proj='utm', zone=str(tile['properties']['zone']), ellps='WGS84')

        #get geometry transformations
        #dt = scenes[0]['properties']['geotrans']
        #dt[0] = ll2utm(*tile['geometry']['coordinates'][0][0])[0]
        #dt[3] = ll2utm(*tile['geometry']['coordinates'][0][2])[1]
        dt = tile['properties']['geotrans']
        print ('tile trans',tile['properties']['geotrans'])


        WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        tile_srs = tile['properties']['proj4']

        utm_proj = pyproj.Proj(tile_srs)
        wgs_proj = pyproj.Proj(WGS84)

        projection_func = partial(pyproj.transform, wgs_proj, utm_proj)
        utm_polys = [transform(projection_func, int_poly) for int_poly in int_polys]
        pix_polys = []

        for utm_poly in utm_polys:

            pix_poly = geometry.Polygon([self.lonlat2pixXY(c,dt) for c in list(utm_poly.exterior.coords)])
            pix_polys.append(pix_poly)

        return pix_polys


    def gen_imgs(self):

        fig, axs = plt.subplots(len(self.select_points),5,figsize=(40,8*len(self.select_points)))

        for ii_p,pp in enumerate(self.select_points):
            pt = self.points[pp]

            for ii_k, (kk,vv) in enumerate(self.products.items()):

                tile = raster_client.dltile_from_latlon(pt['lat'], pt['lon'], self.resolution[kk], self.tilesize[kk], pad=self.tilesize[kk])


                try: #try to load from dist
                    load_str = '_'.join([str(pp),tile['properties']['key'],kk])
                    arr = np.load(os.path.join('data','fig_1',load_str+'_arr.npz'))['arr']
                    meta = json.load(open(os.path.join('data','fig_1',load_str+'_meta.json')))
                    logging.info(f'loading ii_k: {ii_k},ii_p: {ii_p},kk: {kk}')

                except Exception as e:  # if not, download
                    logging.info(f'{e}')
                    logging.info(f'downloading ii_k: {ii_k},ii_p: {ii_p},kk: {kk}')

                    if ii_p==1 and kk=='S2':
                        scenes = metadata_client.search(vv,
                            geom=raster_client.dltile(tile['properties']['key']),
                            start_datetime='2018-12-01',
                            end_datetime=self.edate[kk],
                            #cloud_fraction=0.2,
                            limit=15)['features']
                    elif ii_p==1 and kk=='S2_pred_map':
                        scenes = metadata_client.search(vv,
                            geom=raster_client.dltile(tile['properties']['key']),
                            start_datetime=self.sdate[kk],
                            end_datetime=self.edate[kk],
                            #cloud_fraction=0.2,
                            limit=15)['features'][1:]

                    else:
                        scenes = metadata_client.search(vv,
                            geom=raster_client.dltile(tile['properties']['key']),
                            start_datetime=self.sdate[kk],
                            end_datetime=self.edate[kk],
                            #cloud_fraction=0.2,
                            limit=15)['features']

                    logging.info(f'n scenes: {len(scenes)}')



                    if kk in ['S2','SPOT']:
                        scenes = sorted(scenes, key=lambda k: k['properties']['cloud_fraction'])

                        fill_fraction  = 0
                        arr = np.zeros((self.tilesize[kk]*3, self.tilesize[kk]*3,3))
                        alpha2d = np.zeros((self.tilesize[kk]*3, self.tilesize[kk]*3))
                        ii_w = 0

                        while fill_fraction<0.9:
                            #print ('ii_w', ii_w)


                            new_arr, meta = raster_client.ndarray(
                                                scenes[ii_w]['id'],
                                                bands=self.bands[kk],
                                                scales=self.scales[kk],
                                                ot='UInt16',
                                                dltile=tile['properties']['key'],
                                                processing_level='surface'
                                            )

                            alpha = np.moveaxis(np.stack([alpha2d>0]*3),0,-1)
                            #print (alpha.shape, arr.shape, new_arr.shape)

                            arr[alpha==0]=new_arr[:,:,0:3][alpha==0]


                            alpha2d += new_arr[:,:,3]

                            fill_fraction = np.sum(alpha2d>0)/alpha2d.shape[0]/alpha2d.shape[1]

                            #print (fill_fraction,np.max(arr), np.min(arr), arr.shape)
                            ii_w+=1

                    elif kk in ['SPOT_pred_map']:


                        arr = np.zeros((self.tilesize[kk]*3, self.tilesize[kk]*3))

                        for s in scenes:
                            #print ('ii_w', ii_w)


                            new_arr, meta = raster_client.ndarray(
                                            s['id'],
                                            bands=self.bands[kk],
                                            scales=self.scales[kk],
                                            ot='Byte',
                                            dltile=tile['properties']['key'],
                                            processing_level='surface'
                                        )

                            arr[arr==0]=new_arr[arr==0]

                    else:
                        arr, meta = raster_client.ndarray(
                                            scenes[0]['id'],
                                            bands=self.bands[kk],
                                            scales=self.scales[kk],
                                            ot='Byte',
                                            dltile=tile['properties']['key'],
                                            processing_level='surface'
                                        )

                    load_str = '_'.join([str(pp),tile['properties']['key'],kk])
                    np.savez(os.path.join('data','fig_1',load_str+'_arr.npz'), arr=arr)
                    json.dump(meta,open(os.path.join('data','fig_1',load_str+'_meta.json'),'w'))


                if kk=='S2':
                    axs[ii_p][ii_k].imshow((arr*2.5/255).clip(0,1.))
                elif kk=='SPOT':
                    axs[ii_p][ii_k].imshow((arr/255).clip(0,1.))
                else:
                    axs[ii_p][ii_k].imshow((arr/255).clip(0.,1.), cmap='hot')

                axs[ii_p][ii_k].axis('off')


            pix_polys = self.get_pix_polys(tile)
            #print (tile)

            for poly in pix_polys:
                xs, ys = poly.exterior.xy

                axs[ii_p][4].plot(xs,ys,c='#008c8c',linewidth=3)
                axs[ii_p][4].fill(xs,ys,c='c')
            axs[ii_p][4].set_xlim([0,self.tilesize['SPOT']*3])
            axs[ii_p][4].set_ylim([0,self.tilesize['SPOT']*3])
            axs[ii_p][4].set_xticks([])
            axs[ii_p][4].set_yticks([])
            axs[ii_p][4].invert_yaxis()
            axs[ii_p][4].set_aspect('equal',adjustable='box')
            axs[ii_p][4].text(0.03,0.9,self.ABC[ii_p],fontsize=40,transform=axs[ii_p][4].transAxes)



        axs[0][0].set_title('Sentinel-2 RGB Image',fontsize=36)
        axs[0][1].set_title('SPOT6/7 RGB Image',fontsize=36)
        axs[0][2].set_title('Sentinel-2 Prediction Map',fontsize=36)
        axs[0][3].set_title('SPOT6/7 Prediction Map',fontsize=36)
        axs[0][4].set_title('Vectorised Polygons',fontsize=36)

        #axs[0][0].set_ylabel('A')
        #axs[1][0].set_ylabel('B')
        #axs[2][0].set_ylabel('C')
        #axs[3][0].set_ylabel('D')
        #axs[4][0].set_ylabel('E')

        plt.tight_layout()
        plt.savefig('./fig-1/outp_'+','.join([str(ii) for ii in self.select_points])+'.png')
        plt.show()









if __name__ =="__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    windows = {
    'GB':[-13,50,4,60]
    }

    if not os.path.exists('fig-1'):
        os.makedirs('fig-1')

    fig_gen = Fig1_generator()
    fig_gen.gen_imgs()


