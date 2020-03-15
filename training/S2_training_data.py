# built-in
import json, re, glob, os, pickle, logging, sys

# packages
import geojson, pyproj
import matplotlib.pyplot as plt
from functools import partial
from random import shuffle
import numpy as np
from shapely import geometry
from shapely.affinity import affine_transform
from shapely.ops import transform
from PIL import Image, ImageDraw
import descarteslabs as dl
from area import area

# project
from utils import *

# conf
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class MakeS2TrainingData:

    def __init__(self):
        logging.info(f'Initialising data handler...')
        self.raster_client = dl.Raster()
        self.metadata_client = dl.Metadata()
        exists_or_mkdir('data/')
        exists_or_mkdir('data/S2_unet/')

        self.trn_dltiles = json.load(open(exists_or_download('data/all_trn_dltiles.geojson'),'r'))['features']
        self.trn_polygons = json.load(open(exists_or_download('data/all_trn_polygons.geojson'),'r'))['features']


    def annotation_from_tile(self,tile_key,ii_t,mode='trn'):
        logging.info(f'Fetching tile {tile_key}')
        tile = self.raster_client.dltile(tile_key)

        # get a random season
        season = np.random.choice([0,1,2,3])

        season_start = {
            0:'2018-01-01',
            1:'2018-04-01',
            2:'2018-07-01',
            3:'2018-10-01'
        }
        season_end = {
            0:'2018-03-31',
            1:'2018-06-30',
            2:'2018-09-30',
            3:'2018-11-30'
        }

        # get scenes for dltile
        scene_ind=0
        scenes = self.metadata_client.search('sentinel-2:L1C', geom=tile, start_datetime=season_start[season],  end_datetime=season_end[season], cloud_fraction=0.2, limit=15)['features']
        scenes = sorted(scenes, key=lambda k: k.properties.cloud_fraction, reverse=False)
        if not scenes:
            # no scenes
            return None
        scene=scenes[scene_ind]

        # get geometry transformations
        WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        tile_srs = tile["properties"]["proj4"]
        dt = tile['properties']['geotrans']
        dt_shapely = [
                1 / dt[1],
                dt[2],
                dt[4],
                1 / dt[5],
                -dt[0] / dt[1],
                -dt[3] / dt[5],
            ]

        utm_proj = pyproj.Proj(tile_srs)
        wgs_proj = pyproj.Proj(WGS84)

        projection_func = partial(pyproj.transform, wgs_proj, utm_proj)

        tile_poly = geometry.shape(tile['geometry'])

        # get intersecting polygons
        tile_poly = geometry.Polygon(tile['geometry']['coordinates'][0])
        tile_poly_utm = transform(projection_func, tile_poly)

        if mode=='trn':
            all_polygons = [geometry.shape(pp['geometry']) for pp in self.trn_polygons]
        else:
            all_polygons = [geometry.shape(pp['geometry']) for pp in self.test_polygons]

        intersect_polys = [pp for pp in all_polygons if pp.intersects(tile_poly)] #transform(projection_func, pp).intersects(tile_poly_utm)]

        # get array
        fill_frac = 0.
        while (fill_frac<0.8) and (scene_ind<len(scenes)):
            ### get the array data - comes back as bytes [0:255]
            bands = ['red', 'green', 'blue', 'nir', 'red-edge','red-edge-2', 'red-edge-3', 'red-edge-4', 'swir1','swir2','water-vapor','cirrus','coastal-aerosol','alpha']

            tile_arr, meta = self.raster_client.ndarray(
                scene.id, bands=bands, scales=[[0, 10000]] * 14, ot='UInt16', dltile=tile['properties']['key'], processing_level='surface'
            )
            fill_frac = np.sum(tile_arr[:,:,-1]>0)/tile_arr.shape[0]/tile_arr.shape[1]
            if fill_frac<0.8:
                scene_ind+=1
                try:
                    scene = scenes[scene_ind]
                except:
                    # not enough scenes to fill tile
                    return None

        # clip tile arr [0.,1.]
        tile_arr = (tile_arr/255.).clip(0.,1.)


        # make an annotation array
        annotations = np.zeros((tile['properties']['tilesize'], tile['properties']['tilesize'])) #np.ones((arr.shape[0], arr.shape[1]))*128
        im = Image.fromarray(annotations, mode='L')
        draw = ImageDraw.Draw(im)

        # draw annotation polygons
        for pp in intersect_polys:

            pp_utm = transform(projection_func, pp)

            pp_utm_intersection = pp_utm.intersection(tile_poly_utm)

            if pp_utm_intersection.type == 'MultiPolygon':
                sub_utm_geoms = list(pp_utm_intersection)
            else:
                sub_utm_geoms = [pp_utm_intersection]

            for sub_utm_geom in sub_utm_geoms:
                pix_geom = affine_transform(sub_utm_geom, dt_shapely)
                xs, ys = pix_geom.exterior.xy
                draw.polygon(list(zip(xs, ys)), fill=255)

                for hole in sub_utm_geom.interiors:
                    xs,ys = hole.xy
                    draw.polygon(list(zip(xs, ys)), fill=0)


        annotations = np.array(im)

        # output
        features = []
        features.append(geojson.Feature(geometry=tile_poly, properties=tile['properties']) )

        for p in intersect_polys:
            features.append(geojson.Feature(geometry=p, properties={}) )

        fc_out = geojson.FeatureCollection(features)

        logging.info(f'Writing data and annotation for {tile_key}')
        json.dump(fc_out,open('data/S2_unet/'+str(ii_t)+'.geojson','w'))
        np.savez('data/S2_unet/'+str(ii_t)+'.npz', data = tile_arr, annotation=annotations)


        return True

    def make_records(self,directory):
        """
        Makes a Pickle of a list of record dicts storing data and meta information
        """
        npz_files = glob.glob(os.path.join(directory, '*.npz'))
        meta_files = glob.glob(os.path.join(directory,'*.geojson'))

        records = []
        for npz in npz_files:
            ii = npz.split('/')[-1].split('.')[0]
            meta = [m for m in meta_files if m.split('/')[-1].split('.')[0]==ii][0]
            records.append({'data':npz, 'meta':meta})

        shuffle(records)
        pickle.dump(records, open(os.path.join(directory,'records.pickle'),'wb'))



if __name__ == "__main__":

    trn_data = MakeS2TrainingData()
    #shuffle(trn_data.trn_dltiles)
    #for ii_t, tt in enumerate(trn_data.trn_dltiles[0:101]):
    #    trn_data.annotation_from_tile(tt['properties']['key'],ii_t)

    trn_data.make_records(os.path.join(os.getcwd(),'data','S2_unet'))





