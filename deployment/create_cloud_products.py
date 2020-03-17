"""
Create the cloud products for storing pipeline results
"""

# built-in
import logging, sys, os, datetime

# packages
import yaml
import descarteslabs as dl

# conf
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class CreateCloudProducts:

    def __init__(self):
        # create the dl tasks client
        self.catalog = dl.Catalog()

        # load the cloud fn yaml
        self.config = yaml.safe_load(open(os.path.join(os.getcwd(),'cloud_products.yaml'),'r'))

    def create_cloud_products(self,which=None):
        if type(which)==list:
            # create cloud products from a list
            for prod_key in which:
                self._create_cloud_product(prod_key, self.config[prod_key])


        elif type(which)==str:
            # create a single cloud product
            self._create_cloud_product(which, self.config[which])

        elif not which:
            # create cloud products for each cloud product
            for prod_key, prod_conf in self.config.items():
                self._create_cloud_product(prod_key, prod_conf)

        yaml.dump(self.config,open(os.path.join(os.getcwd(),'cloud_products.yaml'),'w'))

    
    def _create_cloud_product(self, prod_key, prod_conf):
        logging.info(f'Creating DL cloud product {prod_key}...')

        if prod_conf['type']=='vector':
            dl_obj = dl.vectors.FeatureCollection.create(
                                name=prod_conf['name'],
                                title=prod_conf['title'],
                                description=prod_conf['description'])
            logging.info(f'Created FC {dl_obj.id}')
            self.config[prod_key]['cloud_id'] = dl_obj.id

        elif prod_conf['type']=='raster':
            dl_obj = self.catalog.add_product(
                                    prod_conf['name'],
                                    title=prod_conf['title'],
                                    description=prod_conf['description'])
            logging.info(f"Created raster product {dl_obj['data']['id']}")
            for band in prod_conf['other']['bands']:
            
                self.catalog.add_band(
                    dl_obj['data']['id'],
                    name=band['name'],
                    type=band['type'],
                    srcband=band['srcband'],
                    dtype=band['dtype'],
                    nbits=band['nbits'],
                    data_range=band['data_range'],
                    colormap_name=band['colormap_name'])
                logging.info(f"Created band {band['name']}...")

            self.config[prod_key]['cloud_id'] = dl_obj['data']['id']

        self.config[prod_key]['timestamp'] = str(datetime.datetime.now())





if __name__=="__main__":
    ccp = CreateCloudProducts()
    ccp.create_cloud_products()

