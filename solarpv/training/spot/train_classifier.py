"""
Train a ResNet50 classifier to filter solar pv dections
"""
import argparse
from datetime import datetime
import json
import os

from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from optimizer import LRMultiplierAdam

IMAGE_SIZE = 224

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation
BATCH_SIZE_TESTING = 1

# Parameters for this model training
params = {
    "model_name": "solar_pv_v5.3",
    # DATASET METADATA #
    "global_metadata": {
        "local_ground": "classifier_ground/",
        "local_model": "./",
        "n_pos": 3621,  # whole set
        "n_neg": 3621,
    },
    "model": {
        "kwargs": {
            "include_top": False,
            "pooling": "avg",
            "weights": "imagenet",
            "num_classes": 2,
            "activation": "softmax",
            "freeze_weights": False,
        },
    },
    # TRAINING METADATA #
    # Metadata to define the training stage
    "training_kwargs": {
        "train_dir": "classifier_ground/train/",
        "val_dir": "classifier_ground/val/",
        "batchsize_train": 50,
        "batchsize_val": 50,
        # 'epochs': 100,
        "epochs": 2,
        "initial_epoch": 0,
        "optimizer": LRMultiplierAdam(multipliers={"dense_1": 10.0}, lr=1e-4),
        "loss": "categorical_crossentropy",
    },
}


def ResNet50_binary(
    include_top=False,
    pooling="avg",
    weights="imagenet",
    num_classes=2,
    activation="softmax",
    freeze_weights=True,
):
    """
    Define a ResNet50 binary classifier model.

    Defaults are set for a 'softmax', 2 class binary classifier.

    Returns
    -------
    model : keras model
    """
    model = Sequential()

    # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    model.add(ResNet50(include_top=include_top, pooling=pooling, weights=weights))

    # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
    model.add(Dense(num_classes, activation=activation))

    # Say not to train first layer (ResNet) model as it is already trained
    if freeze_weights:
        model.layers[0].trainable = False

    return model


def train(params=params,):
    """
    Train a model
    """
    # Parse the parameter file and extract relevant variables
    ptrain = params["training_kwargs"]  # Convenient pointer

    # Set output files
    model_fname = os.path.join(
        params["global_metadata"]["local_model"], params["model_name"] + ".hdf5"
    )
    history_fname = os.path.join(
        params["global_metadata"]["local_model"], params["model_name"] + "_history.pkl"
    )
    tensorboard_dir = os.path.join(params["global_metadata"]["local_model"], "tb")
    loss_csv = os.path.join(
        params["global_metadata"]["local_model"], params["model_name"] + "_loss.csv"
    )

    # Load and compile the model
    model = ResNet50_binary(**params["model"]["kwargs"])
    sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=sgd, loss=params["training_kwargs"]["loss"], metrics=["accuracy"]
    )

    # preprocessing_function is applied on each image but only after re-sizing & augmentation
    # (resize => augment => pre-process)
    # Each of the keras.application.resnet* preprocess_input MOSTLY mean BATCH NORMALIZATION (applied on each batch)
    # stabilize the inputs to nonlinear activation functions
    # Batch Normalization helps in faster convergence
    data_generator = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.9, 1.2),
        preprocessing_function=preprocess_input,
    )

    # flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)
    # Both train & valid folders must have NUM_CLASSES sub-folders
    train_generator = data_generator.flow_from_directory(
        ptrain["train_dir"],
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=ptrain["batchsize_train"],
        class_mode="categorical",
    )
    n_train_batches = len(train_generator)

    validation_generator = data_generator.flow_from_directory(
        ptrain["val_dir"],
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=ptrain["batchsize_val"],
        class_mode="categorical",
    )
    n_val_batches = len(validation_generator)

    print("Batch size info:")
    print(
        ptrain["batchsize_train"],
        len(train_generator),
        ptrain["batchsize_val"],
        len(validation_generator),
    )

    # Set callbacks
    # Callback 1: early stopping
    cb_early_stopper = EarlyStopping(monitor="val_loss", patience=15, verbose=1)

    # Callback 2: model checkpoint
    cb_checkpointer = ModelCheckpoint(
        filepath=model_fname, monitor="val_loss", save_best_only=True, mode="auto"
    )
    # Callback 3: log file to save train/val loss per epoch
    append = ptrain["initial_epoch"] > 0
    csv_logger = CSVLogger(loss_csv, append=append)

    # Callback 4: dynamically reduce loss rate
    lr_callback = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=10, min_lr=1e-7, verbose=1
    )

    # Callback 5: tensorboard
    log_dir = "{0}/{1}".format(tensorboard_dir, datetime.now())
    log_dir = log_dir.replace(" ", "__")
    os.makedirs(log_dir, exist_ok=True)
    print("Tensorboard Logs going to {0}".format(log_dir))
    tensorboard = TensorBoard(log_dir=log_dir)

    callbacks_list = [
        cb_checkpointer,
        cb_early_stopper,
        csv_logger,
        lr_callback,
        tensorboard,
    ]

    # Check with user
    print(model.summary())
    print(params)
    print("Save final model to: {}".format(model_fname))
    _ = input("Hit any key to continue")

    # Train the model!
    fit_steps = (
        ptrain["steps_per_epoch"] if "steps_per_epoch" in ptrain else n_train_batches
    )
    val_steps = (
        ptrain["validation_steps"] if "validation_steps" in ptrain else n_val_batches
    )
    fit_history = model.fit_generator(
        train_generator,
        steps_per_epoch=fit_steps,
        epochs=ptrain["epochs"],
        validation_data=validation_generator,
        validation_steps=val_steps,
        callbacks=callbacks_list,
    )
    print("Done Training. File written out to: {}".format(model_fname))

    # Save history
    save_history(fit_history.history, history_fname)

    return model


def save_history(history, fname):
    """
    history dict. E.g., fit_history.history
    """
    # Convert all lists to picklable float datatypes
    for kk in history.keys():
        tmp = [float(val) for val in history[kk]]
        history[kk] = tmp

    with open(fname, "w") as ff:
        json.dump(history, ff)


def load_history(history_fname):
    with open(history_fname, "r") as ff:
        history = json.loads(ff.read())
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.train:
        train(params)
