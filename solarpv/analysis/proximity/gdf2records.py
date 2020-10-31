import pygeos, time, pyproj, json, geojson, os
import pandas as pd
import geopandas as gpd
gpd.options.use_pygeos=True


# pd.set_option('mode.chained_assignment',None)
from tqdm import tqdm
tqdm.pandas()

from shapely import geometry, ops, wkt
from functools import partial

from solarpv.utils import get_utm_zone, V_inv

gdf = pd.DataFrame(gpd.read_file('./data/final_mini.gpkg'))

PROJ_WGS = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")


N_PARTITIONS = 8

BUFFER = 10000 #m 

def wgs_buffer_to_file(row):
	pt = row['geometry'].representative_point()

	utm_zone = get_utm_zone(pt.x,pt.y)

	PROJ_UTM = pyproj.Proj(proj='utm',zone=utm_zone,ellps='WGS84')

	wgs2utm = partial(pyproj.transform, PROJ_WGS, PROJ_UTM)
	utm2wgs = partial(pyproj.transform, PROJ_UTM, PROJ_WGS)

	shp_utm = ops.transform(wgs2utm, row['geometry'])

	shp_utm_buffer = shp_utm.buffer(BUFFER) #m

	shp_wgs_buffer = ops.transform(utm2wgs, shp_utm_buffer)

	fname = f'{row["unique_id"]}_{str(BUFFER)}.geojson'

	gj = geojson.FeatureCollection([geojson.Feature(geometry=shp_wgs_buffer, properties={})])
	json.dump(gj, open(os.path.join(os.getcwd(),'data','landmark',fname),'w'))


gdf.progress_apply(lambda row: wgs_buffer_to_file(row), axis=1)