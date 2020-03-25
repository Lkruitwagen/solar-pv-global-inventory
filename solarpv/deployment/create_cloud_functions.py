"""
Create the cloud functions for the pipeline
"""
# built-in
import logging, sys, os, datetime

# packages
import yaml
import descarteslabs as dl

# lib
from deployment.cloud_dl_functions import DL_CLOUD_FUNCTIONS

# conf
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class CreateCloudFunctions:

    def __init__(self):
        # create the dl tasks client
        self.tasks = dl.Tasks()

        # load the cloud fn yaml
        self.config = yaml.safe_load(open(os.path.join(os.getcwd(),'cloud_functions.yaml'),'r'))

    def create_cloud_functions(self,which=None):
        if type(which)==list:
            # create cloud functions from a list
            for fn_key in which:
                self._create_cloud_function(fn_key, self.config[fn_key])


        elif type(which)==str:
            # create a single cloud function
            self._create_cloud_function(which, self.config[which])

        elif not which:
            # create cloud functions for each cloud function
            for fn_key, fn_conf in self.config.items():
                self._create_cloud_function(fn_key, fn_conf)

        yaml.dump(self.config,open(os.path.join(os.getcwd(),'cloud_functions.yaml'),'w'))

    
    def _create_cloud_function(self, fn_key, fn_conf):
        logging.info(f'Creating DL cloud function {fn_key}...')
        fn = self.tasks.create_function(
            DL_CLOUD_FUNCTIONS[fn_key],
            image=fn_conf['image'],
            name=fn_conf['name'],
            requirements=fn_conf['requirements'],
            maximum_concurrency=fn_conf['maximum_concurrency'],
            memory=fn_conf['memory'],
            retry_count=fn_conf['retry_count'],
            task_timeout=fn_conf['task_timeout'],
            )

        self.config[fn_key]['group_id'] = fn.group_id
        self.config[fn_key]['timestamp'] = str(datetime.datetime.now())





if __name__=="__main__":
    ccf = CreateCloudFunctions()
    ccf.create_cloud_functions()
