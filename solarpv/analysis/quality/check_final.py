import requests, json, io, os
import numpy as np
from shapely import geometry
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm
tqdm.pandas()

from solarpv.utils import V_inv

import geopandas as gpd
import pandas as pd

API_KEY = open(os.path.join(os.getcwd(),'maps_api_key.json'),'r').read()

def img_poly(ft_poly,mmpix):

    img_coords = []
    for pt in list(ft_poly.exterior.coords):


        dist, angle, dummy = V_inv((ft_poly.centroid.y,ft_poly.centroid.x),(pt[1],pt[0]))
        dist=dist*1000

        img_coords.append((400+2*(dist/mmpix*np.cos(2.*np.pi*(angle-90.)/360.)),400+2*(dist/mmpix*np.sin(2*np.pi*(angle-90.)/360.))))

    return img_coords

def poly2staticimg(poly_wgs, idx):
    ### take a wgs polygon and grab a static maps snapshot of it

    bbox = poly_wgs.bounds
    centroid = poly_wgs.centroid

    ### This just returns a LxW bbox in m
    box_sides = (V_inv((bbox[1],bbox[0]),(bbox[1],bbox[2]))[0]*1000,
                      V_inv((bbox[1],bbox[0]),(bbox[3],bbox[0]))[0]*1000)


    ### max side len in m
    side_len = np.ceil(max(box_sides))

    ### This dict is the m/pixel in the google maps static basemap
    zoom_dict = dict(zip(range(1,21),[156543.03392 * np.cos(centroid.y * np.pi / 180) / np.power(2, z) for z in range(1,21)]))


    ### chooses the zoom level for us based on side_len
    zoom = np.max(np.argwhere(np.array([(zoom_dict[k]*400-max(box_sides)) for k in range(1,21)])>0.))+1

    pix_poly = img_poly(poly_wgs,zoom_dict[zoom])

    urlstr = ''.join(["""https://maps.googleapis.com/maps/api/staticmap?center=""",
                str(centroid.y)+""","""+str(centroid.x),
                """&zoom="""+str(zoom),
                """&size=400x400&scale=2&maptype=satellite&format=png&key=""", str(API_KEY)])

    r = requests.get(urlstr, allow_redirects=True)


    image_data = r.content
    #image = Image.open(image_data)
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')

    arr = np.asarray(image)

    fig, axs=plt.subplots(1,1,figsize=(24,12))
    axs.imshow(arr)
    xs, ys = geometry.Polygon(pix_poly).exterior.xy
    axs.plot(xs,ys, color='c', linewidth=2.)
    axs.set_title('Google Basemap (indeterminate date), {:.2f}m/px'.format(zoom_dict[zoom]), fontsize=20)
    fig.savefig(os.path.join(os.getcwd(),'data','check_final',f'{idx}.png'))
    plt.close()


def get_osm_samples(gdf_path, N):
    gdf = pd.DataFrame(gpd.read_file(gdf_path))
    #gdf['unique_id'] = range(len(gdf))

    idx = np.random.choice(len(gdf),N, replace=False)

    gdf.iloc[idx,:].progress_apply(lambda row: poly2staticimg(row['geometry'], row['unique_id']), axis=1)


if __name__=="__main__":
    get_osm_samples(os.path.join(os.getcwd(),'data','SPV_vis.gpkg'),200)