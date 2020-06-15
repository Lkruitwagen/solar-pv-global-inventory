import json, re, glob, geojson, os, pickle, logging, sys
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from shapely import geometry
from shapely.affinity import affine_transform
from shapely.ops import transform
from PIL import Image, ImageDraw
import multiprocessing as mp

import descarteslabs as dl
from area import area
import pyproj

from utils import *

logging.info(f'Initialising data handler...')
raster_client = dl.Raster()
metadata_client = dl.Metadata()

trn_dltiles = json.load(open(os.path.join(os.getcwd(),'data','all_trn_dltiles.geojson'),'r'))['features']
trn_polygons = json.load(open(os.path.join(os.getcwd(),'data','all_trn_polygons.geojson'),'r'))['features']


def annotation_from_tile(tile_key,ii_t,mode='trn'):
    print(f'Fetching tile {tile_key}')
    tile = raster_client.dltile(tile_key)

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
    scenes = metadata_client.search('sentinel-2:L1C', geom=tile, start_datetime=season_start[season],  end_datetime=season_end[season], cloud_fraction=0.2, limit=15)['features']
    scenes = sorted(scenes, key=lambda k: k.properties.cloud_fraction, reverse=False)
    if not scenes:
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
        all_polygons = [geometry.shape(pp['geometry']) for pp in trn_polygons]
    else:
        all_polygons = [geometry.shape(pp['geometry']) for pp in test_polygons]

    intersect_polys = [pp for pp in all_polygons if pp.intersects(tile_poly)] #transform(projection_func, pp).intersects(tile_poly_utm)]

    # get array
    fill_frac = 0.
    while (fill_frac<0.8) and (scene_ind<len(scenes)):
        ### get the array data - comes back as bytes [0:255]
        bands = ['red', 'green', 'blue', 'nir', 'red-edge','red-edge-2', 'red-edge-3', 'red-edge-4', 'swir1','swir2','water-vapor','cirrus','coastal-aerosol','alpha']

        tile_arr, meta = raster_client.ndarray(
            scene.id, bands=bands, scales=[[0, 10000]] * 14, ot='UInt16', dltile=tile['properties']['key'], processing_level='surface'
        )
        fill_frac = np.sum(tile_arr[:,:,-1]>0)/tile_arr.shape[0]/tile_arr.shape[1]
        if fill_frac<0.8:
            scene_ind+=1
            scene = scenes[scene_ind]

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
    json.dump(fc_out,open('training/data/S2_unet/'+str(ii_t)+'.geojson','w'))
    np.savez('training/data/S2_unet/'+str(ii_t)+'.npz', data = tile_arr, annotation=annotations)


    return True

def multidownload(n_cpus,keys):
    pool = mp.Pool(n_cpus)
    pool.starmap(annotation_from_tile, list(zip(keys, range(len(keys)))))

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    keys = [tt['properties']['key'] for tt in trn_dltiles[0:12]]
    multidownload(3,keys)




