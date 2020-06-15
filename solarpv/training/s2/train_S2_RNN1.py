 # LSTM1: v26_L8L8CD24D24_1200_932.h5
 # LSTM2: v4_i2L12Dr_nM.h5

"""
Train Sequence Model 1
"""

# built-in
import logging, json, geojson, datetime, sys, os
from random import shuffle

# packages
import numpy as np
from shapely import geometry
from area import area

# local
from solarpv.utils import *

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

print ('gpu',tf.test.is_gpu_available())


#

class TrainS2RNN1:

    def __init__(self, data_gj, outp_fname, val_split=0.7):
        """
        """
        # i/o
        self.gj = json.load(open(data_gj,'r'))
        self.outp_fname = outp_fname

        # const
        self.date_format = "%Y-%m-%d"
        self.baseline = datetime.datetime.strptime('2016-01-01', self.date_format)
        self.input_features = ['X_tseries','X_aspect_ratio', 'X_area', 'X_ratio_perim_area']

        # training parameters
        self.BATCH_SIZE = 300
        self.N_CLASSES = 2
        self.EPOCHS = 300
        self.LEARNING_RATE = 1e-7
        self.TSERIES_SHAPE = (15,2)

        # data prep
        shuffle(self.gj['features'])
        split_int = int(val_split*len(self.gj['features']))

        data_trn = self._prep_data(geojson.FeatureCollection(self.gj['features'][0:split_int]), test=True) #X_trn_tseries,X_trn_aspect_ratio, X_trn_area, X_trn_ratio_perim_area, Y_trn
        data_cv = self._prep_data(geojson.FeatureCollection(self.gj['features'][split_int:]), test=True) #X_cv_tseries,X_cv_aspect_ratio, X_cv_area, X_cv_ratio_perim_area, Y_cv

        self.data = {
            'trn':dict(zip(self.input_features + ['Y'],data_trn)),
            'cv':dict(zip(self.input_features + ['Y'],data_cv)),
        }




    def _try_float(self,ff):
        try:
            float(ff)
            return True
        except:
            return False
    def _parse_properties(self,ft):
        data = {}
        for pp,vv in ft['properties'].items():
            
            if pp.split(':')[0]=='sentinel-2':
                dd = (datetime.datetime.strptime(pp.split(':')[2][0:10], self.date_format) - self.baseline).days
                
                if self._try_float(vv):
                    if dd not in data.keys():
                        data[dd]={}
                    data[dd][pp.split(':')[-1]]=vv

        return np.array([(k,data[k]['M'],data[k]['P']) for k in sorted(data.keys())])


    def _prep_data(self, fc, test=False):
        M = len(fc['features'])
        X_trn_tseries = -1.*np.ones((M,15,2))
        X_trn_aspect_ratio = np.zeros((M,1))
        X_trn_area = np.zeros((M,1))
        X_trn_ratio_perim_area = np.zeros((M,1))

                                    
        for ii_f,ft in enumerate(fc['features']):
            data = self._parse_properties(ft)
            X_trn_tseries[ii_f,(15-data.shape[0]):,:] = data[:,1:3]
            ft_poly = geometry.Polygon(ft['geometry']['coordinates'][0])

            box_coords = np.array(ft_poly.minimum_rotated_rectangle.exterior.coords.xy)
            l = V_inv((box_coords[0][0],box_coords[1][0]),(box_coords[0][1],box_coords[1][1]))[0]*1000 #m
            w = V_inv((box_coords[0][1],box_coords[1][1]),(box_coords[0][2],box_coords[1][2]))[0]*1000 #m

            
            X_trn_aspect_ratio[ii_f,0] = min(l,w)/max(l,w)
            X_trn_area[ii_f,0] = area(ft['geometry'])
            X_trn_ratio_perim_area[ii_f,0] = geometry.Polygon(ft['geometry']['coordinates'][0]).length*1000/area(ft['geometry'])
            
            
        X_trn_ratio_perim_area *= 100
        X_trn_area = (np.log(X_trn_area)/20.).clip(0.,1.)
        
        if not test:
            return [X_trn_tseries,X_trn_aspect_ratio, X_trn_area, X_trn_ratio_perim_area]
        else:
            Y_trn = np.zeros((M,1))
            for ii_f,ft in enumerate(fc['features']):
            
                Y_trn[ii_f,0] = ft['properties']['label']
            
            return [X_trn_tseries,X_trn_aspect_ratio, X_trn_area, X_trn_ratio_perim_area, Y_trn]

    def _model(self,input_shape):

        # inputs: timeseries, area, aspect ratio, perim-area ratio
        input_tseries = Input(shape=input_shape, dtype='float32')
        input_area = Input(shape=(1,), dtype='float32')
        input_aspect_ratio = Input(shape=(1,), dtype='float32')
        input_ratio_perim_area = Input(shape=(1,), dtype='float32')
        
        # LSTM blocks
        X = LSTM(8,return_sequences=True)(input_tseries)
        X = LSTM(8, return_sequences=False)(X)

        concat = Concatenate()([X,input_area, input_aspect_ratio, input_ratio_perim_area])
        

        outp = Dense(24)(concat)
        outp = Activation('relu')(outp)
        outp = Dropout(0.5)(outp)   
        outp = Dense(24)(outp)
        outp = Activation('relu')(outp)
        outp = Dropout(0.5)(outp)
        outp = Dense(1)(outp)
        outp = Activation('sigmoid')(outp)
        
        model = Model([input_tseries, input_area, input_aspect_ratio, input_ratio_perim_area], outp)

        return model

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
            [self.data['trn'][ft] for ft in self.input_features], 
            self.data['trn']['Y'], 
            epochs=self.EPOCHS, 
            batch_size=self.BATCH_SIZE, 
            validation_data = ([self.data['cv'][ft] for ft in self.input_features], self.data['cv']['Y']),
            shuffle=True, 
            verbose=True)

        model.save(self.outp_fname)

if __name__ == "__main__":
    trn = TrainS2RNN1(
        data_gj=os.path.join(os.getcwd(),'data','fts_RNN1.geojson'),
        outp_fname='s2_rnn1.h5')
    trn.train()

    
