"""
Image-Target generator for training this UNet.
"""
# import packages
import numpy as np
import rasterio

from tensorflow import keras


def read_tif(path: str, channels_last: bool = True):
    """
    Load a tiff file into a numpy ndarray.
    
    Parameters
    ----------
    path: str
        File path to tif file you want to read
    channels_last: bool
        The tif file is saved with the channels axis first, however,
        Keras/Tensorflow usually expect channels to be the last axis.
        By default, return the array with the channels axis last.

    Returns
    -------
    image: numpy ndarray
        The image from this tif file. The axis order depends on kwarg CHANNELS_LAST.
    """
    try:
        with rasterio.open(path) as ds:
            image = ds.read()
    except AttributeError as error:
        raise Exception("Could not open: %s" % path) from error

    n_dims = len(image.shape)
    if n_dims < 3:
        image = image[np.newaxis]

    if channels_last:
        # Let's have bands (channels) as the last rather than first axis, as
        # usually expected by Keras/Tensorflow
        return np.transpose(image, axes=(1, 2, 0))

    return image


class DataGenerator(keras.utils.Sequence):
    """
    Keras data generator.
    
    The training data should be saved with images and targets as separate files with the same prefix.
    The sample keys for this generator (e.g., train or val) should be written to a text file, one key per line.
    For example:
    my_data_dir/
        sample_0001_image.tif
        sample_0001_target.tif
        sample_0002_image.tif
        sample_0002_target.tif
    
    my_data_dir/train_keys.txt contains:
        sample_0001
        sample_0002
        
    Augmentations are provided as a list of transformation functions. See `transforms.py`.
    """

    def __init__(
        self,
        data_list,
        batch_size=32,
        dim=(512, 512, 4),
        shuffle=True,
        augment=False,
        transforms=None,
    ):
        """
        Parameters
        ----------
        data_list : path to text file
            Text file with list of keys.
        batch_size : int
            size of batch to be yielded for training
        dim : list
            Training image size: [npix_y, npix_x, bands]
        shuffle : bool
            If true, shuffle the dataset
        augment : bool
            If true, apply augmentations
        transforms : list of functions
            List of augmentation transformations to apply to the yielded samples.
            
        Returns
        -------
        Tuple of (image_batch, target_batch)
        image_batch has shape [batch_size, npix_x, npix_y, bands]
        target_batch has shape [batch_size, npix_x, npix_y, 1]
        
        Example
        -------
        trf = [
               transforms.CastTransform(feature_type='float32', target_type='bool'),
               transforms.SquareImageTransform(),
               transforms.AdditiveNoiseTransform(additive_noise=30.),
               transforms.MultiplicativeNoiseTransform(multiplicative_noise=0.3),
               transforms.NormalizeFeatureTransform(mean=128., std=1.),
               transforms.FlipFeatureTargetTransform(),
              ]
        trn_generator = DataGenerator('path_to_data/train_keys.txt', batch_size=16, dim=(512,512, 4),
                              shuffle=True, augment=True,
                              transforms=trf,
                             )
        """
        self.dim = dim
        self.batch_size = batch_size
        self.data_list = data_list
        self.indexes = None  # int indices (possibly random), set by self.on_epoch_end()
        self.shuffle = shuffle
        self.augment = augment
        self.transforms = transforms

        # read in list of keys
        with open(data_list, "r") as f:
            self.keys = f.read().splitlines()
        self.n_keys = len(self.keys)

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.keys) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        
        Parameters
        ----------
        index : int
            Batch index. If batch size is 10 and index is 3, this will yield samples 30-39
        """
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        batch_keys = [self.keys[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_keys)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.keys))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_keys):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.dim[0], self.dim[1], 1))

        # Generate data
        for i, key in enumerate(batch_keys):
            # load image and target
            f_img = key + "_image.tif"
            f_trg = key + "_target.tif"
            img = read_tif(f_img, channels_last=True)
            trg = read_tif(f_trg)

            if self.augment and self.transforms:
                for transform in self.transforms:
                    img, trg = transform(img, trg)

            X[i, ...] = img
            y[i, ...] = trg

        return X, y
