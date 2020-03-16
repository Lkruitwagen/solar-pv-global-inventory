"""
Train Sequence Model 2
"""

# built-in
import logging, json, geojson, datetime, sys, os
from random import shuffle

# packages
import numpy as np
from shapely import geometry
from area import area
import matplotlib.pyplot as plt

# local
from utils import *

# ML
import tensorflow as tf
from tensorflow.python import keras
from keras.models import Model,load_model, Sequential
from keras.layers import Dense, Input, Dropout, SimpleRNN,LSTM, Activation, Concatenate, TimeDistributed, Reshape, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.utils import to_categorical

# conf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class TrainS2RNN2:

    def __init__(self, data_gj, outp_fname, val_split=0.7):
        # i/o
        self.gj = json.load(open(data_gj,'r'))
        self.outp_fname = outp_fname

        # const
        self.date_format = "%Y-%m-%d"
        self.baseline = datetime.datetime.strptime('2016-01-01', self.date_format)

        # training parameters
        self.BATCH_SIZE = 300
        self.N_CLASSES = 2
        self.EPOCHS = 250
        self.LEARNING_RATE = 1e-7
        self.TSERIES_SHAPE = (160,2)

        # data prep
        shuffle(self.gj['features'])
        split_int = int(val_split*len(self.gj['features']))

        X_trn, Y_trn = self._prep_data(geojson.FeatureCollection(self.gj['features'][0:split_int]), return_labels=True) 
        X_cv, Y_cv = self._prep_data(geojson.FeatureCollection(self.gj['features'][split_int:]), return_labels=True)


        self.data = {
            'trn':{'X':X_trn,'Y':Y_trn},
            'cv':{'X':X_cv,'Y':Y_cv},
        }


    def _parse_props_w_features(self,ft):
        all_data = {}
        for k,v in ft['properties'].items():
            if k not in ['dltile','prediction','primary_id'] and len(v)>3:
                load_string = json.loads(v.replace(',','","').replace('[','["').replace(']','"]').replace('{','{"').replace(':','":').replace(']",','],'))
                for k2,v2 in load_string.items():
                    all_data[k+'-'+k2]=v2
                        
        data_arr = np.zeros((len(all_data.keys()),3))
        for ii_k, k in enumerate(sorted(all_data.keys())):
            data_arr[ii_k,0] = (datetime.datetime.strptime(k,self.date_format)-self.baseline).days
            data_arr[ii_k,1] = np.nan_to_num(float(all_data[k][0]))
            data_arr[ii_k,2] = np.nan_to_num(float(all_data[k][1]))
        
        return data_arr


    def _prep_data(self, fc, return_labels=False):

        X_ts = -1*np.ones((len(fc['features']),160,3))
        Y = np.zeros((len(fc['features'])))

        for ii,ft in enumerate(fc['features']):
            ts  = self._parse_props_w_features(ft)

            X_ts[ii,max(0,(160-ts.shape[0])):,:]= ts[abs(min(0,(160-ts.shape[0]))):,:]

            try:
                Y[ii]=int(ft['properties']['label'])
            except Exception as e:
                logging.exception('error')
                Y[ii]=0


        for ii in range(X_ts.shape[0]):
            for jj in range(160):
                if X_ts[ii,jj,2]>0. and X_ts[ii,jj,1]>0.:
                    X_ts[ii,jj,2] = X_ts[ii,jj,2]/X_ts[ii,jj,1]

        if return_labels:
            return X_ts[:,:,1:], Y
        else:
            return X_ts[:,:,1:]


    def _model(self,input_shape):

        # inputs: timeseries
        input_tseries = Input(shape=input_shape, dtype='float32')
        
        # LSTM blocks
        X = LSTM(12,return_sequences=False)(input_tseries)
        
        outp = Dense(1)(X)
        outp = Activation('sigmoid')(outp)
        
        model = Model(input_tseries, outp)

        return model

    def _post_visualise(self, N=10):

        Y_pred = self.model.predict(self.data['cv']['X'])

        for ii in range(N):

            fig, ax = plt.subplots(1,1,figsize=(10,10))
            ax.scatter(range(160),self.data['cv']['X'][ii,:,0], c='g')
            ax.scatter(range(160),self.data['cv']['X'][ii,:,1], c='b')
            ax.text(0,0,'label: {}, prediction: {}'.format(self.data['cv']['Y'][ii],Y_pred[ii]))
            plt.show()

    def train(self):

        # Load the model
        model = self._model(self.TSERIES_SHAPE)

        # compile the model
        model.compile(
            optimizer='adam', 
            loss='binary_crossentropy',
            metrics=['accuracy'])
        logging.info(model.summary())

        history = model.fit(
            self.data['trn']['X'], 
            self.data['trn']['Y'], 
            epochs=self.EPOCHS, 
            batch_size=self.BATCH_SIZE, 
            validation_data = (self.data['cv']['X'], self.data['cv']['Y']),
            shuffle=True, 
            verbose=True)

        model.save(self.outp_fname)
        self.model = model

if __name__ == "__main__":
    trn = TrainS2RNN2(
        data_gj=os.path.join(os.getcwd(),'data','fts_RNN2.geojson'),
        outp_fname='s2_rnn2.h5')
    trn.train()
    #trn._post_visualise()

    
