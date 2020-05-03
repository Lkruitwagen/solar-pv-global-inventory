"""
Functions to train a Solar_pv model.

This entry point can be run from a schema document.

Requirements: tf 1.14
"""
import argparse
from datetime import datetime
from functools import partial
import inspect
import logging
import os
from time import time
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    CSVLogger,
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)

from generator import DataGenerator
from unet import UNet
import transforms

params = {
    "seed": 21,  # for train/val data split
    # Training data specifications
    # DATASET METADATA #
    "data_metadata": {
        "products": ["airbus:oneatlas:spot:v2"],
        "bands": ["red", "green", "blue", "nir"],
        "resolution": 1.5,
        "start_datetime": "2016-01-01",
        "end_datetime": "2018-12-31",
        "tilesize": 512,
        "pad": 0,
    },
    # GLOBAL METADATA #
    "global_metadata": {
        "local_ground": "ground/",  # directory containing image-target pairs
        "local_model": "model/",  # directory to write this model
    },
    # MODEL METADATA
    "model_name": "solar_pv_airbus_spot_rgbn_v5",
    # TRAINING METADATA #
    # Metadata to define the training stage
    "training_kwargs": {
        "datalist": "train_keys.txt",
        "batchsize": 16,
        "val_datalist": "val_keys.txt",
        "val_batchsize": 16,
        "epochs": 150,
        "image_dim": (512, 512, 4),  # This is the size of the training images
    },
    "transforms": [
        transforms.CastTransform(feature_type="float32", target_type="float32"),
        transforms.SquareImageTransform(),
        transforms.AdditiveNoiseTransform(additive_noise=30.0),
        transforms.MultiplicativeNoiseTransform(multiplicative_noise=0.3),
        transforms.NormalizeFeatureTransform(mean=128.0, std=1.0),
        transforms.FlipFeatureTargetTransform(),
    ],
}


def binary_crossentropy(y_true, y_pred, small=1e-3, weight=1.0, sample_weight=None):
    """Compute the (weighted) binary crossentropy"""
    bce = weight * y_true * tf.math.log(y_pred + small) + (1 - y_true) * tf.math.log(
        1 - y_pred + small
    )

    return -tf.math.reduce_mean(bce)


def assert_consistency(model, shape=(16, 512, 512, 4)):
    """ Checks whether a model gives self-consistent results
    by feeding it all ones and making sure the outputs match."""
    img = np.ones(shape).astype("float")
    f = model.predict(img)
    is_self_consistent = np.all(f[0] == f[-1])
    assert is_self_consistent


def return_model_file_paths(output_dir, model_name):
    model_fname = os.path.join(output_dir, "{}.hdf5".format(model_name))
    checkpoint_stub = os.path.join(output_dir, "checkpoint_{}".format(model_name))
    loss_fname = os.path.join(output_dir, "loss_{}.csv".format(model_name))

    return model_fname, checkpoint_stub, loss_fname


def train_from_document(params=params):
    """
    Train a Building model.

    All relevant parameters should be defined in a parameter file. See params dictionary above.

    Parameters:
    ----------
    document : dictionary
        See above for an example

    Returns
    ----------
    Nothing.

    Saves
    ----------
    model : .hdf5 file
        Final Keras model
    loss : CSF file
        Saves csv file with loss as a function of epoch to file `loss_fname`
    logfile : text file
        File of logging statements
    """
    t0 = time()

    data_path = params["global_metadata"]["local_ground"]
    print("*** Train model from data in {}".format(data_path))

    # Get a few convenient pointers for clean code
    kw_train = params["training_kwargs"]
    model_name = params["model_name"]

    if not os.path.exists(data_path):
        print(
            "working dir {} doesn't exist. You need to get your training data!".format(
                data_path
            )
        )

    # ---------------------------
    # Define the output files
    # ---------------------------
    local_output_dir = params["global_metadata"]["local_model"]
    os.makedirs(local_output_dir, exist_ok=True)

    # Log file
    logfile = os.path.join(local_output_dir, "train_" + model_name + ".log")
    print("Logging output to %s" % logfile)
    logger = logging.getLogger("solar_logger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(logging.FileHandler(filename=logfile))
    logger.propagate = False
    logger.info("Start train: %s" % str(datetime.now()))

    # Log input arguments
    args, varargs, keywords, mylocals = inspect.getargvalues(inspect.currentframe())
    logger.info("Runtime parameters:")
    logger.info(inspect.formatargvalues(args, varargs, keywords, mylocals))

    # Output file names
    model_fname, checkpoint_stub, loss_fname = return_model_file_paths(
        local_output_dir, model_name
    )

    logger.info("model: %s" % model_fname)
    logger.info("checkpoint_stub: %s" % checkpoint_stub)
    logger.info("loss: %s" % loss_fname)

    # ---------------------------
    # Define the model
    # ---------------------------
    model = UNet(n_bands=4, n_classes=2)

    # Get the loss function
    loss = partial(binary_crossentropy, weight=4.0)
    loss.__name__ = "binary_crossentropy"  # required by Keras

    # Get the optimizer (usually Adam)
    optimizer = Adam(lr=1e-3)

    # Compile the Keras model
    model.compile(optimizer=optimizer, loss=loss)

    tilesize = params["data_metadata"]["tilesize"]
    assert_consistency(
        model,
        shape=(
            kw_train["batchsize"],
            tilesize,
            tilesize,
            len(params["data_metadata"]["bands"]),
        ),
    )

    # Write model summary to log file
    model.summary(print_fn=lambda x: logger.info(x))

    # ---------------------------
    # Define the generator
    # ---------------------------
    train_keys_file = os.path.join(
        params["global_metadata"]["local_ground"], kw_train["datalist"]
    )
    val_keys_file = os.path.join(
        params["global_metadata"]["local_ground"], kw_train["val_datalist"]
    )

    trn_generator = DataGenerator(
        train_keys_file,
        batch_size=kw_train["batchsize"],
        dim=kw_train["image_dim"],
        shuffle=True,
        augment=True,
        transforms=params["transforms"],
    )

    val_generator = DataGenerator(
        val_keys_file,
        batch_size=kw_train["val_batchsize"],
        dim=kw_train["image_dim"],
        shuffle=True,
        augment=True,
        transforms=params["transforms"],
    )
    n_train = trn_generator.n_keys
    n_val = val_generator.n_keys

    # If we didn't find any training or validation data, return with an error
    if (n_train == 0) or (n_val == 0):
        raise ValueError(
            "Missing train/validation data: ",
            "n_train = %d, n_val = %d. path = %s." % (n_train, n_val, data_path),
        )

    logger.info("n_train = %d, n_val = %d" % (n_train, n_val))

    # ---------------------------
    # Setup callbacks
    # ---------------------------

    # Callback 1: checkpoints to save the weights file
    checkpoint_fname = checkpoint_stub + "_{epoch:03d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        checkpoint_fname,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
        period=2,
    )
    checkpoint.set_model(model)

    # Callback 2: log file to save train/val loss per epoch
    csv_logger = CSVLogger(loss_fname)

    # Callback 3: dynamically reduce loss rate
    lr_callback = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=10, min_lr=1e-7, verbose=1
    )

    # Callback 4: Stop early if loss has plateaued.
    # In cases with very little training data, we may still want to remove this and over-fit.
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=15, verbose=1
    )

    # Callback 5: tensorboard
    log_dir = "{0}/{1}".format(local_output_dir, datetime.now())
    log_dir = log_dir.replace(" ", "__")
    os.makedirs(log_dir, exist_ok=True)
    print("Tensorboard Logs going to {0}".format(log_dir))
    logger.info("Saving tensorboard logs to {0}".format(log_dir))
    tensorboard = TensorBoard(log_dir=log_dir)

    callbacks_list = [lr_callback, checkpoint, csv_logger, early_stopping, tensorboard]

    # ---------------------------
    # Fit model
    # ---------------------------

    # We can't use batches that are larger than the dataset
    if kw_train["batchsize"] > n_train:
        warnings.warn(
            "batch size {} bigger than training set {}. "
            "Setting batchsize=n_train.".format(kw_train["batchsize"], n_train)
        )
        kw_train["batchsize"] = n_train
    if kw_train["val_batchsize"] > n_val:
        warnings.warn(
            "batch size {} bigger than val set {}. "
            "Setting batchsize=n_val.".format(kw_train["val_batchsize"], n_val)
        )
        kw_train["val_batchsize"] = n_val

    fit_steps = (
        kw_train["steps_per_epoch"]
        if "steps_per_epoch" in kw_train
        else n_train // kw_train["batchsize"]
    )
    val_steps = (
        kw_train["validation_steps"]
        if "validation_steps" in kw_train
        else n_val // kw_train["batchsize"]
    )
    print("steps_per_epoch = {}, validation_steps = {}".format(fit_steps, val_steps))
    print("epochs = {}".format(kw_train["epochs"]))
    model.fit_generator(
        generator=trn_generator,
        validation_data=val_generator,
        verbose=1,
        steps_per_epoch=fit_steps,
        validation_steps=val_steps,
        epochs=kw_train["epochs"],
        callbacks=callbacks_list,
    )

    print("Finished training Model. Save to {}".format(model_fname))
    # Save the model.
    # We save the base model, not the multi-gpu hybrid.
    logger.info("Save final model to: %s" % model_fname)
    model.save(model_fname)

    logger.info("Finish train: %s" % str(datetime.now()))
    t1 = time() - t0
    print("That took %f seconds" % t1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Execute training")
    args = parser.parse_args()

    if args.train:
        print("*** training")
        train_from_document()

    t1 = datetime.now()
    print("Finish train.py: {}".format(t1))
