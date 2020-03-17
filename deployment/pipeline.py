"""
Run the pipeline
"""

# built-in
import os, sys, logging, datetime, json

# packages
import yaml
import descarteslabs as dl
import shapely.geometry as geometry

# lib
from deployment.cloud_dl_functions import DL_CLOUD_FUNCTIONS

# conf
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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


def get_shp(shp_str):
    """
    Return a shapely geometry in WGS84 lon/lat 
    input: shp_str - a string corresponding to an iso-3166-1 or -2 administrative area for admin-level 1 (countries) and -2 (states/provinces) respectively
    """

    if len(shp_str.split('-'))>1:
        load_fts = json.load(open(os.path.join(os.getcwd(),'data','ne_10m_admin_1_states_provinces.geojson'),'r'))
        select_fts = [ft for ft in load_fts['features'] if ft['properties']['iso_3166_2']==shp_str]
    else:
        load_fts = json.load(open(os.path.join(os.getcwd(),'data','ne_10m_admin_0_countries.geojson'),'r'))
        select_fts = [ft for ft in load_fts['features'] if ft['properties']['ISO_A2']==shp_str]
    all_shps = [geometry.shape(ft['geometry']) for ft in select_fts]

    return reduce2mp(flatten_polys(all_shps))

def shp_exclusions(shp, shp_str):
    pop_json = json.load(open(os.path.join(os.getcwd(),'data','popshp_gt1_d7k.geojson'),'r')) # main population dilation shape
    dnr_json = json.load(open(os.path.join(os.getcwd(),'data','do_not_run.geojson'),'r')) # removes some census-based null areas in Canada and Australia

    pop_shps = []
    for ii_f, ft in enumerate(pop_json['features']):
        #print (ii_f)
        try:
            pop_shps.append(geometry.shape(ft['geometry']))
        except:
            pass

    dnr_shps = [geometry.shape(ft['geometry']) for ft in dnr_json['features']]


    pop_unions = []
    for ii_s,pop_shp in enumerate(pop_shps):
        if not pop_shp.intersection(shp).is_empty:
            pop_unions.append(pop_shp.intersection(shp))

    deployment_shp = reduce2mp(pop_unions)

    for shp in dnr_shps:
        deployment_shp = deployment_shp.difference(shp)


    ak_poly = geometry.Polygon([[-169,0],[-169,60],[-141,60],[-141,0]])
    clip_poly = geometry.Polygon([[-180,0],[-180,60],[179,60],[179,0]])
    if shp_str =='US-AK':
        deployment_shp = deployment_shp.intersection(ak_poly).buffer(0)
    elif shp_str[0:2] in ['CA','RU']:
        deployment_shp = deployment_shp.intersection(clip_poly).buffer(0)

    if deployment_shp.is_empty:
        logging.error('empty geom!')
        return None

    return deployment_shp


class Pipeline:

    def __init__(self):
        self.raster_client = dl.Raster()
        self.catalog_client = dl.Catalog()
        self.tasks_client = dl.Tasks()

        self.fn_config = yaml.safe_load(open(os.path.join(os.getcwd(),'cloud_functions.yaml'),'r'))
        self.prod_config = yaml.safe_load(open(os.path.join(os.getcwd(),'cloud_products.yaml'),'r'))

    def run(self,which,shp_str):

        if type(which)==list:
            # create cloud functions from a list
            for fn_key in which:
                self._run_cloud_function(fn_key, shp_str)

        elif type(which)==str:
            # create a single cloud function
            self._run_cloud_function(which, shp_str)

        elif not which:
            # create cloud functions for each cloud function
            for fn_key, fn_conf in self.fn_config.items():
                self._run_cloud_function(fn_key, shp_str)

    def _run_cloud_function(self,fn_key, shp_str):

        shp = get_shp(shp_str)
        shp = shp_exclusions(shp, shp_str)

        async_function = self.tasks_client.get_function(name=self.fn_config[fn_key]['name'])

        logging.info(f"Running {self.fn_config[fn_key]['name']} for {shp_str}")

        if fn_key=='S2Infer1':

            sdate='2019-09-30'
            edate='2019-12-31'

            tiles = self.raster_client.dltiles_from_shape(
                self.fn_config[fn_key]['tiledict']['resolution'], 
                self.fn_config[fn_key]['tiledict']['tilesize'], 
                self.fn_config[fn_key]['tiledict']['pad'],
                geometry.mapping(shp))


            logging.info(f"Running {len(tiles['features'])} tiles and storing to raster {self.prod_config['S2-R1-Primary']['cloud_id']} and vector {self.prod_config['S2-V1-Primary']['cloud_id']}")

            done_scenes, null = dl.scenes.search(aoi=geometry.mapping(shp), products=[self.prod_config['S2-R1-Primary']['cloud_id']])
            done_keys = [ft['key'].split(':')[3].replace('_',':') for ft in done_scenes]
            run_tiles = [t for t in tiles['features'] if t.properties.key not in done_keys]
            
            logging.info(f"Found {len(done_keys)} previously-run tiles, running remaining {len(run_tiles)} tiles")

            for ii_t, tile in enumerate(run_tiles):

                async_function(
                    dltile=tile, 
                    src_product_id='sentinel-2:L1C', 
                    dest_product_id=self.prod_config['S2-R1-Primary']['cloud_id'], 
                    fc_id=self.prod_config['S2-V1-Primary']['cloud_id'], 
                    sdate=sdate,
                    edate=edate, 
                    run_ii=ii_t)
                if ii_t % 10 ==0:  
                    logging.info(f'Running tile {ii_t} of {len(run_tiles)}')

        elif fn_key=='S2RNN1':

            tiles = self.raster_client.dltiles_from_shape(
                self.fn_config[fn_key]['tiledict']['resolution'], 
                self.fn_config[fn_key]['tiledict']['tilesize'], 
                self.fn_config[fn_key]['tiledict']['pad'],
                geometry.mapping(shp))


            logging.info(f"Running {len(tiles['features'])} tiles and storing to raster {self.prod_config['S2-R2-Secondary']['cloud_id']}")
            logging.info(f"Input vector:{self.prod_config['S2-V1-Primary']['cloud_id']}, output vector: {self.prod_config['S2-V2-Secondary']['cloud_id']}")

            done_scenes, null = dl.scenes.search(aoi=geometry.mapping(shp), products=[self.prod_config['S2-R2-Secondary']['cloud_id']])
            done_keys = [ft['key'].split(':')[3].replace('_',':') for ft in done_scenes]
            run_tiles = [t for t in tiles['features'] if t.properties.key not in done_keys]
            
            logging.info(f"Found {len(done_keys)} previously-run tiles, running remaining {len(run_tiles)} tiles")

            for ii_t, tile in enumerate(run_tiles):

                async_function(
                    dltile=tile, 
                    src_vector_id=self.prod_config['S2-V1-Primary']['cloud_id'], 
                    dest_vector_id=self.prod_config['S2-V2-Secondary']['cloud_id'], 
                    dest_product_id=self.prod_config['S2-R2-Secondary']['cloud_id'], 
                    push_rast = True
                    )
                if ii_t % 10 ==0:  
                    logging.info(f'Running tile {ii_t} of {len(run_tiles)}')

        elif fn_key=='S2Infer2':
            THRESHOLD=0.5

            fc_src = dl_local.vectors.FeatureCollection(self.prod_config['S2-V2-Secondary']['cloud_id'])
            fc_dest = dl_local.vectors.FeatureCollection(self.prod_config['S2-V3-Deepstack']['cloud_id'])

            logging.info(f"Gathering features passing RNN-1 with threshold {THRESHOLD} and running them through the full imagery stack.")


            all_deep_fts = [f for f in fc_dest.filter(shp).features()]
            logging.info(f"Features already run: {len(all_deep_fts)}")
            deep_ft_ids = [f.properties.primary_id for f in all_deep_fts]

            THRESHOLD = 0.5
            sec_fts = [f for f in fc_src.filter(shp).filter(properties=(dl_p_local.prediction >=THRESHOLD)).features()]
            logging.info(f'Features in geography meeting threshold {THRESHOLD}: {len(sec_fts)}')

            deploy_fts = [f for f in sec_fts if f.properties.primary_id not in deep_ft_ids]
            logging.info(f'Features in geography to deploy: {len(deploy_fts)}')

            for ii_f, f in enumerate(deploy_fts):

                # make a lowpass area filter
                f_area = area(json.dumps(geometry.mapping(f.geometry)).replace('(','[').replace(')',']'))

                if f_area >=80:

                    try:
                        async_function(
                            storage_key=None, 
                            dl_ft=json.dumps(f.geojson), 
                            src_product_id='sentinel-2:L1C', 
                            dest_fc_id=self.prod_config['S2-V3-Deepstack']['cloud_id'], 
                            storage_flag=False
                            )
                        logging.info(f'Doing {ii_f}, p: {ii_f/len(deploy_fts)}, area:{f_area}')
                    except Exception as e:
                        logging.error(e)
                        storage_key = 'TMP_FT_'+f.properties['primary_id']

                        storage_local.set(storage_key,json.dumps(f.geojson),storage_type='data')

                        async_function(
                            storage_key=storage_key, 
                            dl_ft=None, 
                            src_product_id='sentinel-2:L1C', 
                            dest_fc_id=self.prod_config['S2-V3-Deepstack']['cloud_id'], 
                            storage_flag=True
                            )
                        logging.info(f'Doing {ii_f} via storage, key: {storage_key}, p: {ii_f/len(deploy_fts)}, area:{f_area}')


        elif fn_key=='SPOTVectoriser':
            
            # get SPOT scenes
            SPOT_scenes = []
            if shp.type=='Polygon':
                new_scenes, null = [f for f in dl.scenes.search(aoi=geometry.mapping(shp), products=[self.prod_config['SPOT-R1-Primary']['cloud_id']])]
                SPOT_scenes += new_scenes
            else:
                for subshp in list(shp):
                    new_scenes, null = [f for f in dl.scenes.search(aoi=geometry.mapping(subshp), products=[self.prod_config['SPOT-R1-Primary']['cloud_id']])]
                    SPOT_scenes += new_scenes

            logging.info(f"Retrieved {len(SPOT_scenes)} SPOT scenes for the geography")
            scenes_polys = [geometry.shape(s['geometry']) for s in SPOT_scenes]

            if len(scenes_polys)<1:
                logging.error('No scenes!')
                exit()

            scenes_mp = reduce2mp(scenes_polys)

            tiles = self.raster_client.dltiles_from_shape(
                self.fn_config[fn_key]['tiledict']['resolution'], 
                self.fn_config[fn_key]['tiledict']['tilesize'], 
                self.fn_config[fn_key]['tiledict']['pad'],
                geometry.mapping(scenes_mp))

            logging.info(f"Retrieved {len(tiles['features'])} for processing.")

            for tile in tiles['features']:
                async_function(
                    dltile=tile, 
                    src_product_id=self.prod_config['SPOT-R1-Primary']['cloud_id'], 
                    band_names=[self.prod_config['SPOT-R1-Primary']['other']['bands'][0]['name']], 
                    scales=[self.prod_config['SPOT-R1-Primary']['other']['bands'][0]['data_range']], 
                    dest_fc_id=self.prod_config['SPOT-V1-Vecotrised']['cloud_id'],
                    shp_str=shp_str)


if __name__ =="__main__":
    pl = Pipeline()
    pl.run('S2Infer1','BE')