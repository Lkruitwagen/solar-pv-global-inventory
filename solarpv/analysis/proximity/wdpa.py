import pygeos, time, pyproj
import pandas as pd
import geopandas as gpd
gpd.options.use_pygeos=True
import dask.dataframe as dd
#from dask.multiprocessing import get

from shapely import geometry, ops, wkt
from functools import partial

from solarpv.utils import get_utm_zone, V_inv

PROJ_WGS = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

TIC = time.time()

N_PARTITIONS = 4

BUFFER = 5000 #m 

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
    adf.loc[:,'geometry'] = adf.loc[:,'geometry'].apply(lambda el: el.wkt)

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
    idx_pp = [[pp,row['WDPA_PID'][ii],*ops.nearest_points(row['geometry'], pp),1] for ii,pp in enumerate(row['WDPA_geoms']) if row['shp_wgs_buffer'].intersects(pp)]
    
    for record in idx_pp:
        if record[0].intersects(row['geometry']):
            record[-1]=-1
        record.append(V_inv((record[2].x, record[2].y),(record[3].x, record[3].y))[0]*record[-1]) # [pp_geom, WDPA_ID, pt1, pt2, -1/1, km]
        
            
    idx_pp = sorted(idx_pp, key=lambda record: record[-1])
    
    return [[r[1],r[-1]] for r in idx_pp] # WDAP_ID, dist_km



def mixin_WDPA(df, colidx, fpath):


    WDPA = gpd.read_file(fpath)    
    print (f'{colidx}, Constructing tree and getting querying...', end='')
    df.loc[:,'geometry'] = df.loc[:,'geometry'].apply(wkt.loads)
    df.loc[:,'shp_wgs_buffer_pygeos'] = df.loc[:,'shp_wgs_buffer'].apply(pygeos.io.from_wkt)

    print (df['shp_wgs_buffer_pygeos'])
    print (type(df.loc['shp_wgs_buffer_pygeos'][0]))
    
    tree = pygeos.STRtree([pygeos.io.from_shapely(pp) for pp in WDPA.geometry])
    
    Q = tree.query_bulk(df['shp_wgs_buffer_pygeos'].values.tolist())# , predicate='intersects')  # intersects not rapid.
    
    Q_df = pd.DataFrame(Q.T, columns=['df','wdpa'])
    
    print (time.time()-TIC)
    
    print (f'{colidx}, Get WDPA ids and geoms', end='')

    # slooowwww single threading...
    df['WDPA_PID'] = df.apply(lambda row: WDPA.loc[Q_df.loc[Q_df['df']==row.name,'wdpa'].values,'WDPA_PID'].values, axis=1)
    df['WDPA_geoms'] = df.apply(lambda row: WDPA.loc[WDPA['WDPA_PID'].isin(row['WDPA_PID']),'geometry'].values, axis=1)
    print (time.time()-TIC)

    ddf = dd.from_pandas(df, npartitions=N_PARTITIONS) # lol and then ver memory explodey

    df['WDPA_'+str(colidx)] = ddf.apply(lambda row: dist_sort(row), meta=('WDPA_'+str(colidx),object), axis=1).compute()    # MULTI

    WDPA = WDPA.set_index('WDPA_PID')

    def get_WDPA_names(ll):
        # ll => [idx, dist]
        
        ll_idx = [el[0] for el in ll]
        
        NAME_DESIG = WDPA.loc[ll_idx,['NAME','DESIG_ENG']].values.tolist()
        
        for ii in range(len(ll)):
            
            ll[ii] = tuple(ll[ii] + NAME_DESIG[ii])
            
        return ll


    df['WDPA_'+str(colidx)] = df['WDPA_'+str(colidx)].apply(get_WDPA_names)

    return df



    


if __name__=="__main__":

    print ('Loading spv data...')
    df = pd.DataFrame(gpd.read_file('./data/final_mini.gpkg'))


    print ('prepping df (series)', end='')
    
    df.loc[:,'pt'] = df.loc[:,'geometry'].apply(lambda el: el.representative_point())
    df.loc[:,'utm_zone'] = df.loc[:,'pt'].apply(lambda row: get_utm_zone(row.y, row.x))
    df.loc[:,'utm_zone'] = df.loc[:,'utm_zone'].astype(int)
    df = pd.DataFrame(df)
    df.loc[:,'geometry'] = df.loc[:,'geometry'].apply(lambda el: el.wkt).astype(str)
    #print ('type',df['geometry'].type)
    print (time.time()-TIC)

    print ('getting meta')

    meta = prep_df_async(df.iloc[0:5])

    print ('meta',meta)


    print ('casting to dask and running ops...', end='')
    ddf = dd.from_pandas(df, npartitions=N_PARTITIONS)

    df = ddf.map_partitions(prep_df_async, meta=meta).compute()
    print (time.time()-TIC)


    fs = ['./data/WDPA/WDPA_0.gpkg','./data/WDPA/WDPA_1.gpkg','./data/WDPA/WDPA_2.gpkg']

    for colidx, f in enumerate(fs):
        df = mixin_WDPA(df, colidx, f)
        print (df['WDPA_'+str(colidx)])

    print ('DONE!')
    print (df)

    drop_cols = ['WDPA_PID',
                'WDPA_goems',
                'pt',
                'utm_zone',
                'utm_proj_obj',
                'reproj_wgs_utm',
                'reproj_utm_wgs',
                'shp_utm',
                'shp_utm_buffer',
                'shp_wgs_buffer',
                'shp_wgs_buffer_pygeos']