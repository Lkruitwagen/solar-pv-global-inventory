import pygeos, time, pyproj, json
import pandas as pd
import geopandas as gpd
gpd.options.use_pygeos=True
import dask.dataframe as dd
#from dask.multiprocessing import get
pd.set_option('mode.chained_assignment',None)
from tqdm import tqdm
tqdm.pandas()

from shapely import geometry, ops, wkt
from functools import partial

from solarpv.utils import get_utm_zone, V_inv

PROJ_WGS = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

TIC = time.time()

N_PARTITIONS = 8

BUFFER = 10000 #m 

def prep_df_async(adf):
    
    adf.loc[:,'geometry'] = adf.loc[:,'geometry'].apply(wkt.loads)
    
    
    adf.loc[:,'utm_proj_obj'] = adf.apply(lambda el: pyproj.Proj(proj='utm',zone=el['utm_zone'],ellps='WGS84'), axis=1)
    
    adf.loc[:,'reproj_wgs_utm'] = adf.apply(lambda el: partial(pyproj.transform, PROJ_WGS, el['utm_proj_obj']), axis=1)
    
    adf.loc[:,'reproj_utm_wgs'] = adf.apply(lambda el: partial(pyproj.transform, el['utm_proj_obj'], PROJ_WGS), axis=1)
    
    adf.loc[:,'shp_utm'] = adf.apply(lambda row: ops.transform(row['reproj_wgs_utm'], row['geometry']), axis=1)
    
    adf.loc[:,'shp_utm_buffer'] = adf.apply(lambda row: row['shp_utm'].buffer(BUFFER), axis=1)
    
    adf.loc[:,'shp_wgs_buffer'] = adf.apply(lambda row: ops.transform(row['reproj_utm_wgs'], row['shp_utm_buffer']), axis=1)

    # return geometries to strings
    adf.loc[:,'shp_wgs_buffer'] = adf.loc[:,'shp_wgs_buffer'].apply(lambda el: el.wkt).astype(str)
    adf.loc[:,'geometry'] = adf.loc[:,'geometry'].apply(lambda el: el.wkt).astype(str)

    adf.drop(columns=['shp_utm_buffer','shp_utm','reproj_wgs_utm','reproj_utm_wgs','utm_proj_obj'], inplace=True)
    
    #adf['shp_wgs_buffer_pygeos'] = adf.apply(lambda row: pygeos.io.from_shapely(row['shp_wgs_buffer']), axis=1)
    

    return adf

"""
def prep_df_async(adf):
    
    adf['utm_proj_obj'] = adf.apply(lambda el: pyproj.Proj(proj='utm',zone=el['utm_zone'],ellps='WGS84'), meta=('utm_proj_obj',object), axis=1)
    
    adf['reproj_wgs_utm'] = adf.apply(lambda el: partial(pyproj.transform, PROJ_WGS, el['utm_proj_obj']), meta=('reproj_wgs_utm',object), axis=1)
    
    adf['reproj_utm_wgs'] = adf.apply(lambda el: partial(pyproj.transform, el['utm_proj_obj'], PROJ_WGS), meta=('reproj_utm_wgs',object), axis=1)
    
    adf['shp_utm'] = adf.apply(lambda row: ops.transform(row['reproj_wgs_utm'], row['geometry']), meta=('shp_utm',object), axis=1)
    
    adf['shp_utm_buffer'] = adf.apply(lambda row: row['shp_utm'].buffer(BUFFER), meta=('shp_utm_buffer',object), axis=1)
    
    adf['shp_wgs_buffer'] = adf.apply(lambda row: ops.transform(row['reproj_utm_wgs'], row['shp_utm_buffer']), meta=('shp_wgs_buffer',object), axis=1)
    
    adf['shp_wgs_buffer_pygeos'] = adf.apply(lambda row: pygeos.io.from_shapely(row['shp_wgs_buffer']), meta=('shp_wgs_buffer_pygeos',object), axis=1)
    

    return adf
"""

def dist_sort(row):
    idx_pp = [[pp,row['WDPA_PID'][ii],*ops.nearest_points(row['geometry'], pp),1] for ii,pp in enumerate(row['WDPA_geoms']) if wkt.loads(row['shp_wgs_buffer']).intersects(pp)]
    
    for record in idx_pp:
        if record[0].intersects(row['geometry']):
            record[-1]=-1
        record.append(V_inv((record[2].x, record[2].y),(record[3].x, record[3].y))[0]*record[-1]) # [pp_geom, WDPA_ID, pt1, pt2, -1/1, km]
        
            
    idx_pp = sorted(idx_pp, key=lambda record: record[-1])
    
    return [[r[1],r[-1]] for r in idx_pp] # WDAP_ID, dist_km

def dist_sort_async(adf, colidx):

    adf['WDPA_'+str(colidx)] = adf.apply(lambda row: dist_sort(row), axis=1)

    return adf


def mixin_WDPA(df, colidx, fpath):


    print (f'loading WDPA {fpath}')

    WDPA = gpd.read_file(fpath)
    WDPA = WDPA[['WDPA_PID','NAME','DESIG_ENG','geometry']]


    print ('types')

    print (f'{colidx}, Constructing tree and getting querying...', end='')

    # print (df['shp_wgs_buffer_pygeos'])
    # print (type(df['shp_wgs_buffer_pygeos'][0]))
    
    tree = pygeos.STRtree([pygeos.io.from_shapely(pp) for pp in WDPA.geometry])
    
    Q = tree.query_bulk(df['shp_wgs_buffer_pygeos'].values.tolist())# , predicate='intersects')  # intersects not rapid.
    
    Q_df = pd.DataFrame(Q.T, columns=['df','wdpa'])
    
    print (time.time()-TIC)
    
    print (f'{colidx}, Get WDPA ids and geoms', end='')

    # slooowwww single threading...
    df['WDPA_PID'] = df.apply(lambda row: WDPA.loc[Q_df.loc[Q_df['df']==row.name,'wdpa'].values,'WDPA_PID'].values, axis=1)
    df['WDPA_geoms'] = df.apply(lambda row: WDPA.loc[WDPA['WDPA_PID'].isin(row['WDPA_PID']),'geometry'].values, axis=1)
    print (time.time()-TIC)

    print ('Running single thread apply...')
    df['WDPA_'+str(colidx)] = df.progress_apply(lambda row: dist_sort(row), axis=1)    # MULTI

    WDPA = WDPA.set_index('WDPA_PID')

    def get_WDPA_names(ll):
        # ll => [idx, dist]
        
        ll_idx = [el[0] for el in ll]
        
        NAME_DESIG = WDPA.loc[ll_idx,['NAME','DESIG_ENG']].values.tolist()
        
        for ii in range(len(ll)):
            
            ll[ii] = tuple(ll[ii] + NAME_DESIG[ii])
            
        return ll

    print ('getting names')
    df['WDPA_'+str(colidx)] = df['WDPA_'+str(colidx)].progress_apply(get_WDPA_names)

    #df['geometry'] = df['geometry'].apply(lambda el: el.wkt).astype(str)

    return df




def mixin_WDPA_dask(df, colidx, fpath):



    print ('types')
    print (df['geometry'].apply(type).unique())
    print (f'{colidx}, Constructing tree and getting querying...', end='')

    df['geometry'] = df['geometry'].apply(wkt.loads)
    df['shp_wgs_buffer_pygeos'] = df['shp_wgs_buffer'].apply(pygeos.io.from_wkt)

    # print (df['shp_wgs_buffer_pygeos'])
    # print (type(df['shp_wgs_buffer_pygeos'][0]))
    
    tree = pygeos.STRtree([pygeos.io.from_shapely(pp) for pp in WDPA.geometry])
    
    Q = tree.query_bulk(df['shp_wgs_buffer_pygeos'].values.tolist())# , predicate='intersects')  # intersects not rapid.
    
    Q_df = pd.DataFrame(Q.T, columns=['df','wdpa'])
    
    print (time.time()-TIC)
    
    print (f'{colidx}, Get WDPA ids and geoms', end='')

    # slooowwww single threading...
    df['WDPA_PID'] = df.apply(lambda row: WDPA.loc[Q_df.loc[Q_df['df']==row.name,'wdpa'].values,'WDPA_PID'].values, axis=1)
    df['WDPA_geoms'] = df.apply(lambda row: WDPA.loc[WDPA['WDPA_PID'].isin(row['WDPA_PID']),'geometry'].values, axis=1)
    print (time.time()-TIC)

    print ('Running map_partitions...')

    ddf = dd.from_pandas(df, npartitions=N_PARTITIONS) # lol and then ver memory explodey

    meta = dist_sort_async(df.iloc[0:5], colidx)

    print ('async meta')
    print (meta)

    df = ddf.map_partitions(dist_sort_async, colidx, meta=meta).compute()

    #print ('Running asyn apply...')
    #df['WDPA_'+str(colidx)] = ddf.apply(lambda row: dist_sort(row), meta=('WDPA_'+str(colidx),object), axis=1).compute()    # MULTI

    WDPA = WDPA.set_index('WDPA_PID')

    def get_WDPA_names(ll):
        # ll => [idx, dist]
        
        ll_idx = [el[0] for el in ll]
        
        NAME_DESIG = WDPA.loc[ll_idx,['NAME','DESIG_ENG']].values.tolist()
        
        for ii in range(len(ll)):
            
            ll[ii] = tuple(ll[ii] + NAME_DESIG[ii])
            
        return ll


    df['WDPA_'+str(colidx)] = df['WDPA_'+str(colidx)].apply(get_WDPA_names)

    df['geometry'] = df['geometry'].apply(lambda el: el.wkt).astype(str)

    return df



    


if __name__=="__main__":

    print ('Loading spv data...')
    #df = pd.DataFrame(gpd.read_file('./data/final_mini.gpkg'))
    df = pd.DataFrame(gpd.read_file('./data/ABCD_finalized.geojson'))#.iloc[0:1000]
    #df = df[['unique_id','geometry']]


    print ('prepping df (series)', end='')
    
    df['pt'] = df['geometry'].apply(lambda el: el.representative_point())
    df['utm_zone'] = df['pt'].apply(lambda row: get_utm_zone(row.y, row.x))
    df['utm_zone'] = df['utm_zone'].astype(int)
    df = pd.DataFrame(df)
    df['geometry'] = df['geometry'].apply(lambda el: el.wkt).astype(str)
    #print ('type',df['geometry'].type)
    print (time.time()-TIC)

    print ('getting meta')

    meta = prep_df_async(df.iloc[0:5])
    print ('got meta')

    # print ('meta',meta)


    print ('casting to dask and running ops...', end='')
    ddf = dd.from_pandas(df, npartitions=N_PARTITIONS)

    df = ddf.map_partitions(prep_df_async, meta=meta).compute()
    print (time.time()-TIC)


    fs = ['./data/WDPA/WDPA_0.gpkg','./data/WDPA/WDPA_1.gpkg','./data/WDPA/WDPA_2.gpkg']

    ### if single threaded:

    df['geometry'] = df['geometry'].apply(wkt.loads)
    df['shp_wgs_buffer_pygeos'] = df['shp_wgs_buffer'].apply(pygeos.io.from_wkt)
    for colidx, f in enumerate(fs):
        df = mixin_WDPA(df, colidx, f)
        #print (df['WDPA_'+str(colidx)])

    ### if using dask:
    #for colidx, f in enumerate(fs):
    #    df = mixin_WDPA_dask(df, colidx, f)
    #    print (df['WDPA_'+str(colidx)])

    print ('DONE!')

    
    drop_cols = ['pt', 'utm_zone', 'shp_wgs_buffer','shp_wgs_buffer_pygeos', 'WDPA_PID', 'WDPA_geoms']
    
    df.drop(columns=drop_cols, inplace=True)

    df['WDPA_proximity'] = df.progress_apply(lambda row: json.dumps(row['WDPA_0'] + row['WDPA_1'] + row['WDPA_2']), axis=1)

    df.drop(columns=['WDPA_0','WDPA_1','WDPA_2'], inplace=True)

    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'])

    gdf.to_file('./data/SPV_wdpa.gpkg',driver='GPKG')