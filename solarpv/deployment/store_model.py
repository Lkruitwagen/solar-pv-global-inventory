"""
"""
import logging, sys, argparse, os

from tensorflow.python import keras
import descarteslabs as dl

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def store_ML_model(model_path, model_name):
    storage_client = dl.Storage()

    # load model
    model = keras.models.load_model(model_path)
    logging.info(model.summary())

    # get JSON spec and store it
    json_str = model.to_json()

    storage_client.set('_'.join(['ML_model','json',model_name]), json_str, storage_type='data')

    # get model weights
    np_weights = model.get_weights()

    # save weight shapes to be able to convert bytestrings back to np arrays
    np_shapes = {}

    # store weights
    for ii, w in enumerate(np_weights):
        logging.info('storing layer {ii}:',w.shape)
        np_shapes[ii]=w.shape
        storage_client.set('_'.join(['ML_model','weights',model_name,str(ii)]), w.tobytes(), storage_type='data')

    # store weight shapes
    storage_client.set('_'.join(['ML_model','shapes',model_name]), json.dumps(np_shapes), storage_type='data')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to the model to be sent to cloud storage", type=str)
    args = parser.parse_args()

    if args.model_path:
        print (args.model_path)
        store_ML_model(args.model_path,os.path.split(args.model_path)[1].split('.')[-1])

    

