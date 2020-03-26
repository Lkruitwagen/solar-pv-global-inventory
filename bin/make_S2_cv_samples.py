import os, logging

from solarpv.training.S2_training_data import *

# conf
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


generator = MakeS2TrainingData(
	tilespath=os.path.join(os.getcwd(),'data','cv_all_tiles.geojson'),
	polyspath=os.path.join(os.getcwd(),'data','cv_all_polys.geojson'),
	outpath=os.path.join(os.getcwd(),'data','crossvalidation','S2_unet'))

generator.download_all_samples(multi=True,n_cpus=4)

generator.make_records(os.path.join(os.getcwd(),'data','crossvalidation','S2_unet'))