Train the solar model.

Examples of the structure of the training data are given, using empty .tif files

Train the UNet like:
```
$ python train.py --train
```
The training data is pairs of .tif files for images and targets.
The train/val keys should be listed in a text file, `train_keys.txt` and `val_keys.txt`

Train classifier like:
```
train_classifier.py --train
```
The training data are .tif files, split into a folder for `train/` and `val/`.
When running this, the stript will pause for the user to review the arguments before starting training.