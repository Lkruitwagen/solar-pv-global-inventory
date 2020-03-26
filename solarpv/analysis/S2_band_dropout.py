"""
S2 band dropout
"""
# built-in
import os, pickle, logging, sys

# packages
from tensorflow.python import keras
from keras.utils import Sequence
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# lib
from solarpv.training.train_S2_unet import DataGenerator


# conf
K.set_image_data_format('channels_last')
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

print ('gpu',tf.test.is_gpu_available())


class BandDropoutS2:

    def __init__(self,records_path, model_path, batch_size):
        self.records = pickle.load(open(records_path,'rb'))
        self.model = keras.models.load_model(model_path)
        self.band_list = ['red', 'green', 'blue', 'nir', 'red-edge','red-edge-2', 'red-edge-3', 'red-edge-4', 'swir1','swir2','water-vapor','cirrus','coastal-aerosol','alpha']
        self.BATCH_SIZE=batch_size


    def loop_inference(self):

        mi = pd.MultiIndex.from_product([range(len(self.records)),self.band_list+['none']],names=['records','bands'])
        df = pd.DataFrame(index=mi, columns=None)

        logging.info(f'Running band dropout...')
        df['band_dropout'] = self._run_inference(perturbation=None)['data']
        df.to_csv(os.path.join(os.getcwd(),'solarpv','analysis','chkpt.csv'))

        for perturbation_magnitude in [0.1,0.2,0.3]:
            df['additive_'+str(perturbation_magnitude)] = self._run_inference(perturbation='additive', perturbation_magnitude=perturbation_magnitude)['data']
            df.to_csv(os.path.join(os.getcwd(),'solarpv','analysis','chkpt.csv'))

        for perturbation_magnitude in [0.1, 0.2, 0.3]:
            df['multiplicative_'+str(perturbation_magnitude)] = self._run_inference(perturbation='multiplicative', perturbation_magnitude=perturbation_magnitude)['data']
            df.to_csv(os.path.join(os.getcwd(),'solarpv','analysis','chkpt.csv'))


        df.to_csv(os.path.join(os.getcwd(),'solarpv','analysis','band_perturbation.csv'))




    def _run_inference(self, perturbation=None, perturbation_magnitude=0.05):

        """
        run inferences to check sensitivity. Check against:
        Band dropout: perturbation=None
        Additive noise: perturbation='additive'
        Multiplicative noise: perturbation='multiplicative'
        perturbation_magnitude: magnitude of perturbation
        """

        mi = pd.MultiIndex.from_product([range(len(self.records)),self.band_list+['none']],names=['records','bands'])
        df = pd.DataFrame(index=mi, columns=['data'])

        if not perturbation:
            perturbation_str = 'dropout'
        else:
            perturbation_str = perturbation + '_' + str(perturbation_magnitude)

        generator = DataGenerator(self.records, 
                                    batch_size=self.BATCH_SIZE, 
                                    dim=(200,200,14), 
                                    n_channels=1,
                                    n_classes=2, 
                                    shuffle=False, 
                                    augment=False)

        logging.info(f'len generator: {len(generator)}')

        ious = np.zeros((len(self.records), len(self.band_list+['none'])))

        for ii_b in range(len(self.band_list+['none'])):

            for ii_g in tqdm(range(len(generator)), desc=f'band: {(self.band_list+["none"])[ii_b]}'):   # ii_batch

                sample, ann = generator.__getitem__(ii_g)

                ann = np.argmax(ann,axis=-1)



                do_sample = sample.copy()

                if ii_b<(len(self.band_list+['none'])-1):
                    if perturbation==None:
                        do_sample[:,:,:,ii_b]=0#do_sample[:,:,ii]+noise

                    elif perturbation=='additive':
                        noise = perturbation_magnitude*np.random.rand(self.BATCH_SIZE*200*200).reshape(self.BATCH_SIZE,200,200) - perturbation_magnitude/2
                        do_sample[:,:,:,ii_b]=do_sample[:,:,:,ii_b]+noise

                    elif perturbation=='multiplicative':
                        noise = np.ones((self.BATCH_SIZE,200,200))+ perturbation_magnitude*np.random.rand(self.BATCH_SIZE*200*200).reshape(self.BATCH_SIZE,200,200) - perturbation_magnitude/2
                        do_sample[:,:,:,ii_b]=do_sample[:,:,:,ii_b]*noise

                do_sample = do_sample.clip(0,1)


                pred = np.squeeze(self.model.predict(do_sample))
                pred = (pred[...,1] - pred[...,0]).clip(0,1)



                ious_batch = np.sum((pred>0.1).astype(int) * ann, axis=(1,2)) / np.sum(((pred>0.1).astype(int) + ann).clip(0,1), axis=(1,2))  #batch size
                

            ious[ii_g*self.BATCH_SIZE:(ii_g+1)*self.BATCH_SIZE,ii_b]=ious_batch

        df.loc[:,'data'] = ious.ravel()
        return df


    def _run_inference_fig(self, perturbation=None, perturbation_magnitude=0.05, examples_path=None, examples_freq=0.05, batch_size=1):

        if examples_path:
            FIG_FLAG = np.random.rand()<examples_freq
        else:
            FIG_FLAG = False

        if FIG_FLAG:
            fig, axs = plt.subplots(4,4,figsize=(20,20))

        for ii in range(sample.shape[-1]):
            do_sample = sample.copy()
            #do_sample[:,:,ii]=0
            #print (do_sample.shape)
            #print (do_sample.mean(axis=0).mean(axis=0))

            if perturbation==None:
                do_sample[:,:,ii]=0#do_sample[:,:,ii]+noise

            elif perturbation=='additive':
                noise = perturbation_magnitude*np.random.rand(200*200).reshape(200,200) - perturbation_magnitude/2
                do_sample[:,:,ii]=do_sample[:,:,ii]+noise

            elif perturbation=='multiplicative':
                noise = np.ones((200,200))+ perturbation_magnitude*np.random.rand(200*200).reshape(200,200) - perturbation_magnitude/2
                do_sample[:,:,ii]=do_sample[:,:,ii]*noise

            do_sample = do_sample.clip(0,1)
            

            pred = np.squeeze(self.model.predict(do_sample[np.newaxis,...]))
            pred = (pred[...,1] - pred[...,0]).clip(0,1)

            iou = np.sum((pred>0.1).astype(int) * ann) / np.sum(((pred>0.1).astype(int) + ann).clip(0,1))
            ious.append(iou)

            if FIG_FLAG:
                axs[ii//4, ii%4].imshow(pred)
                axs[ii//4, ii%4].set_title(f'{self.band_list[ii]} IoU: {iou}',fontsize=12)
                axs[ii//4, ii%4].set_axis_off()

        pred = np.squeeze(self.model.predict(sample[np.newaxis,...]))
        pred = (pred[...,1] - pred[...,0]).clip(0,1)

        iou = np.sum((pred>0.1).astype(int) * ann) / np.sum(((pred>0.1).astype(int) + ann).clip(0,1))
        ious.append(iou)


        if FIG_FLAG:
            axs[3,2].imshow(pred)
            axs[3,2].set_title(f'All bands IoU: {iou}',fontsize=12)
            axs[3,3].imshow(sample[:,:,0:3])
            axs[3,2].set_axis_off()
            axs[3,3].set_axis_off()
            #logging.info(f'saving fig {str(ii_r)+perturbation_str+".png"}')
            fig.savefig(os.path.join(examples_path,str(ii_r)+'_'+perturbation_str+'.png'))
            fig = None

        df.loc[ii_r,:] = ious
        return df










if __name__=="__main__":
    bdo = BandDropoutS2(
        records_path=os.path.join(os.getcwd(),'data','crossvalidation','S2_unet','records.pickle'),
        model_path=os.path.join(os.getcwd(),'data','S2_unet.h5'),
        batch_size=1)
    bdo.loop_inference()
