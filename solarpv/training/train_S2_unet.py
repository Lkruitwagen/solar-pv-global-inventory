# Train S2-UNET

# built-in
import pickle, copy, logging, os, sys

# packages
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ML
import tensorflow as tf
from tensorflow.python import keras
from keras.layers import Input, Dense, Activation, Cropping2D, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, UpSampling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model, to_categorical, Sequence
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from random import shuffle
from keras.callbacks import CSVLogger, Callback, ModelCheckpoint
import keras.backend as K

# conf
K.set_image_data_format('channels_last')
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

print ('gpu',tf.test.is_gpu_available())


#
class HistPlot(Callback):
    
    def __init__(self, validation_generator, outfile='metrics_hist'):
        self.validation_generator = validation_generator
        self.batch_size = validation_generator.batch_size
        self.outfile = outfile
        #x,y = self.validation_generator.__getitem__(0)
        #print ('nit shapes', x.shape, y.shape)
        
    def on_epoch_end(self, epoch, logs={}):
        
        #print (dir(model))
        print (logs)

        Y = np.zeros((200,200,200,2))
        X = np.zeros((200,200,200,14))
        for ii in range(10):
            #print (ii)
            
            x,y = self.validation_generator.__getitem__(ii)
            X[ii*self.batch_size:(ii+1)*self.batch_size,:,:,:] = x
            Y[ii*self.batch_size:(ii+1)*self.batch_size,:,:,:] = y
            #print ('xy shape', x.shape, y.shape)
        #print (X.shape, Y.shape)
        Y_pred = self.model.predict_generator(self.validation_generator, steps=20)
        
        
                
        ### gen hist plot
        if epoch >0:
            print (self.model.history.history)
            epochs = np.arange(len(self.model.history.history['val_loss']))

            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(epochs, self.model.history.history['loss'], label='loss', color='red')
            ax1.plot(epochs, self.model.history.history['val_loss'], label='val_loss', color='orange')
            ax1.set_ylabel('loss')
            
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width, box.height*0.9])

            ax2 = ax1.twinx()
            ax2.plot(epochs, self.model.history.history['categorical_accuracy'], label='cat_accuracy', color='blue')
            ax2.plot(epochs, self.model.history.history['val_categorical_accuracy'], label='val_cat_accuracy', color='green')
            ax2.set_position([box.x0, box.y0, box.width, box.height*0.9])
            
            h1,l1 = ax1.get_legend_handles_labels()
            h2,l2 = ax2.get_legend_handles_labels()
            
            
            ax1.legend(handles = h1+h2, labels=l1+l2, loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=4)


            plt.savefig(self.outfile+'.png')
            plt.close() 

class DataGenerator(Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, augment=False):
        """ Initialisation """
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augment=augment



    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate indexes of the batch"""
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples of shape:
        (n_samples, *dim, n_channels)
        """

        # Initialization
        X = np.zeros((self.batch_size, *self.dim))
        y = np.zeros((self.batch_size, self.dim[0], self.dim[1]))#, dtype=int)


        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample
            if self.augment:
                k_rot=np.random.choice([0,1,2,3])
                k_flip = np.random.choice([0,1])
                
                X[i,:,:,:] = np.flip(np.rot90(np.load(ID['data'])['data'], k_rot, axes=(0,1)),k_flip)
                y[i,:,:] = np.flip(np.rot90(np.load(ID['data'])['annotation'].astype('int')/255, k_rot, axes=(0,1)),k_flip)

            else:
            
                X[i,:,:,:] = np.load(ID['data'])['data']

                # Store class
                y[i,:,:] = np.load(ID['data'])['annotation'].astype('int')/255
                #print (y)


        return X, to_categorical(y, num_classes=self.n_classes)


class TrainS2Unet:

    def __init__(self, data_dir, outp_fname, trn_records_pickle, val_records_pickle=None):
        """
        """
        # i/o
        self.data_dir = data_dir
        self.outp_fname = outp_fname

        # training parameters
        self.BATCH_SIZE = 2
        self.N_CLASSES = 2
        self.EPOCHS = 40
        self.LEARNING_RATE = 1e-7
        self.INPUT_SHAPE = (200,200,14)

        # data records
        self.trn_records = pickle.load(open(trn_records_pickle,'rb'))
        self.crossval_split = 0.7

        if not val_records_pickle:
            val_indexes = np.random.choice(len(self.trn_records),int((1-self.crossval_split)*len(self.trn_records)))
            self.val_records = [rr for ii,rr in enumerate(self.trn_records) if ii in val_indexes]
            self.trn_records = [rr for ii,rr in enumerate(self.trn_records) if ii not in val_indexes]
        else:
            self.val_records = pickle.load(open(val_records,'rb'))

    def train(self, model_def_file):

        # Load the model
        with open(model_def_file) as f:
            model = model_from_json(f.read())

        # compile the model
        optimizer = Adam(lr=self.LEARNING_RATE)
        model.compile(
            optimizer=optimizer, 
            loss='binary_crossentropy',
            metrics=[categorical_accuracy])
        logging.info(model.summary())

        # instantiate generators 
        trn_generator = DataGenerator(self.trn_records, 
                                batch_size=self.BATCH_SIZE, 
                                dim=self.INPUT_SHAPE, 
                                n_channels=1,
                                n_classes=self.N_CLASSES, 
                                shuffle=True,
                                augment=True)

        val_generator = DataGenerator(self.val_records, 
                                batch_size=self.BATCH_SIZE, 
                                dim=self.INPUT_SHAPE, 
                                n_channels=1,
                                n_classes=self.N_CLASSES, 
                                shuffle=True,
                                augment=False)

        # instantiate callbacks
        csv_cb = CSVLogger('log.csv', append=True, separator=';')
        hist_plot_cb = HistPlot(validation_generator=val_generator, outfile='trn_bootstrap_metrics')
        chkpt_cb = ModelCheckpoint('model_progress.h5', period=5)

        history = model.fit_generator(
                                generator=trn_generator, 
                                validation_data=val_generator, 
                                verbose=1,  
                                epochs=self.EPOCHS, 
                                callbacks=[csv_cb, hist_plot_cb,chkpt_cb])

        model.save(self.outp_fname)

if __name__ == "__main__":
    trn = TrainS2Unet(
        data_dir=os.path.join(os.getcwd(),'data','S2_unet'),
        outp_fname='s2_unet.h5',
        trn_records_pickle=os.path.join(os.getcwd(),'data','S2_unet','records.pickle'))
    trn.train(os.path.join(os.getcwd(),'training','model_resunet.json'))

    
