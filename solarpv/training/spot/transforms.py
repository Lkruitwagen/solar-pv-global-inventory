"""
This file provides a set of common transform functions for generators
that inherit from DlGen.

Transforms are coded as objects with a callable method, with parameters specified when you instantiate a class object.
To use transforms, create an object of the desired class, passing in the keyword arguments to the constructor.

We generally use "feature" to mean the input of a model since "input" is a
reserved python global.

All transformations are applied serially.

Example #1 of applying a transform to an image:
>>> import numpy as np
>>> from appsci_utils.generator.transforms import CastTransform
>>> my_image = np.ones((10,10), dtype='uint8') # dummy image
>>> my_transform = CastTransform(feature_type='float32', target_type='bool')
>>> transformed_img, _ = my_transform(my_image)

Example #2 of using these transforms with a DLGen object:
>>> from appsci_utils.generator.transforms import CastTransform
>>> kwargs = {'feature_type'='float32', 'target_type'='bool'}
>>> my_transform = CastTransform(**kwargs)
>>> gen = ImageTargetGenerator(...)
>>> iter_ = gen.get_iter(..., transforms=[my_transform])

"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage.interpolation import zoom


class TypeSignature:
    """Keep track of both the datatype and range of data values which are valid
    for an application. Transforms may change data types and ranges, and can
    tell a model whether it is receiving the data range it expects.

    The data range doesn't necessarily have to be a hard range. It might
    represent standard deviations, for instance. In this case, we expect
    datarange=(mean - stddev, mean + stddev) and rangetype='stddev'.
    """

    def __init__(self, datatype=None, datarange=None, rangetype="minmax"):
        if datatype is None:
            # allow any
            self.datatype = type
        else:
            if not (isinstance(datatype, np.generic) or isinstance(datatype, type)):
                raise ValueError("Expected a numpy or python type for data")
            self.datatype = datatype

        if datarange is None:
            # allow any
            self.datarange = np.array((-np.inf, np.inf))
        else:
            self.datarange = np.array(datarange).astype(self.datatype)

        if rangetype not in ["minmax", "stddev"]:
            raise ValueError(
                'Expected rangetype to be "minmax" or "stddev", got %s' % rangetype
            )
        self.rangetype = rangetype

    def __repr__(self):
        if self.rangetype == "minmax":
            return "typesig(%s <= %s <= %s)" % (
                self.datarange[0],
                self.datatype.__name__,
                self.datarange[1],
            )
        elif self.rangetype == "stddev":
            return "typesig(%s ~ %s ~ %s)" % (
                self.datarange[0],
                self.datatype.__name__,
                self.datarange[1],
            )

    def __eq__(self, other):
        return (
            (self.datatype == other.datatype)
            and (self.datarange == other.datarange).all()
            and (self.rangetype == other.rangetype)
        )


class Transform(ABC):
    """The parameter 'function' can be provided to apply a transform assuming
    a single feature and/or single target apply to multi-feature,
    multi-target data. Commonly used functions are available as static
    methods to this Transform class."""

    params = []  # List names of expected parameters, or dict with defaults
    multiplier = 1  # Output count multiplier (>1 means multiout transform)

    @staticmethod
    def apply_features_targets(f, features, targets):
        """Apply the transform to each feature,target pair in order.
        Must have the same numbers of features and targets per sample."""
        new_features, new_targets = [], []
        for feature, target in zip(features, targets):
            new_feature, new_target = f(feature, target)
            new_features.append(new_feature)
            new_targets.append(new_target)
        return new_features, new_targets

    @staticmethod
    def apply_features(f, features, targets):
        """Apply the transform to each feature in a multi-feature sample.
        target is passed through."""
        new_features = []
        for feature in features:
            new_feature, _ = f(feature, targets)
            new_features.append(new_feature)
        return new_features, targets

    @staticmethod
    def apply_targets(f, features, targets):
        """Apply the transform to each target in a multi-target sample.
        feature is passed through."""
        new_targets = []
        for target in targets:
            _, new_target = f(features, target)
            new_targets.append(new_target)
        return features, new_targets

    def __init__(self, function=None, **params):
        if isinstance(function, str):
            self._function = getattr(self, function)
        elif function is None:
            self._function = lambda f, features, targets: f(features, targets)
        else:
            self._function = function

        # Set defaults
        if isinstance(self.params, dict):
            for param, value in self.params.items():
                setattr(self, param, value)
        else:
            for param in self.params:
                setattr(self, param, None)

        # Save given parameter values
        for param, value in params.items():
            if param not in self.params:
                raise ValueError("Got unexpected parameter %s" % param)
            setattr(self, param, value)

    @abstractmethod
    def _call(self, feature, target=None):
        """Override in subclass to apply a transformation"""
        return feature, target

    def transform_typesignature(self, feature_typesig, target_typesig=None):
        """Override in subclass if the transformation changes the datatype or datarange.
        Many transforms will not need to change this. Note that feature_typesig can
        also be None when no input type is specified."""
        return feature_typesig, target_typesig

    def __call__(self, feature, target=None):
        # For maintaining behavior of function-only transforms, we return
        # the multiplier when no feature is given. This can be changed if
        # the function-only transforms are to be removed.
        if feature is None:
            return self.multiplier
        new_feature, new_target = self._function(self._call, feature, target)
        # Guarantee that if feature or target were given as None, that we return None
        return (
            None if feature is None else new_feature,
            None if target is None else new_target,
        )


def symmetries_of_the_square(array, flip_indices=None):
    """Transform an array according to the 8 symmetries of the square.

    Parameters
    ----------
        array : numpy array shape [row, col, ...]
        flip_indices [int or list of ints]    indices of flip transformations to return
    Returns
    -------
      1) if flip_indices is a list or None, returns a list of the result when
         an array is flipped and rotated in every which way.
      2) if flip_indices in an integer, returns one transform of the array.
    """
    # The default is a data augmentation which returns 7 transformations
    # (the original orientation is returned by _augment())
    if flip_indices is None:
        flip_indices = np.arange(1, 8)

    if isinstance(flip_indices, int):
        flip_indices = [flip_indices]

    arrays = []
    if 0 in flip_indices:
        # Original orientation
        arrays.append(array)
    if 1 in flip_indices:
        # Diagonal flip 1
        arrays.append(array.swapaxes(0, 1))
    if 2 in flip_indices:
        # Vertical flip [::-1,...]
        arrays.append(np.flipud(array))
    if 3 in flip_indices:
        # Diagonal flip 2
        arrays.append(np.fliplr(np.rot90(array, k=1)))
    if 4 in flip_indices:
        # Horizontal flip [:,::-1,...]
        arrays.append(np.fliplr(array))
    if 5 in flip_indices:
        # 90deg rotation
        arrays.append(np.rot90(array, k=1))
    if 6 in flip_indices:
        # 180deg rotation
        arrays.append(np.rot90(array, k=2))
    if 7 in flip_indices:
        # 270deg rotation
        arrays.append(np.rot90(array, k=3))

    # if only one orientation was requested, return just the array (not a list)
    if len(arrays) == 1:
        arrays = arrays[0]

    return arrays


# Registrable transforms and multiout_transforms
class NormalizeFeatureTransform(Transform):
    params = ["mean", "std"]

    def _call(self, feature, target=None):
        if self.mean is None:
            mean = np.mean(feature)
        else:
            mean = self.mean
        if self.std is None:
            std = np.sqrt(np.mean(feature ** 2))
        else:
            std = self.std

        return (feature - mean) / std, target

    def transform_typesignature(self, feature_typesig, target_typesig=None):
        if feature_typesig is None:
            return None, target_typesig

        if self.mean is None:
            raise ValueError(
                "Cannot know data range ahead of time without parameter 'mean'"
            )
        if self.std is None:
            raise ValueError(
                "Cannot know data range ahead of time without parameter 'mean'"
            )

        # Set the datarange to +/- 1 and rangetype=stddev to reflect how the data now looks
        new_feature_typesig = TypeSignature(
            datatype=feature_typesig.datatype,
            datarange=np.array((-1, 1), dtype=feature_typesig.datatype),
            rangetype="stddev",
        )
        return new_feature_typesig, target_typesig


class CastTransform(Transform):
    params = ["feature_type", "target_type"]

    def _call(self, feature, target=None):
        if self.feature_type is not None:
            if type(feature) in [np.ndarray, np.ma.core.MaskedArray]:
                feature = feature.astype(self.feature_type)
            else:
                feature = self.feature_type(feature)

        if target is not None and self.target_type is not None:
            if type(target) in [np.ndarray, np.ma.core.MaskedArray]:
                target = target.astype(self.target_type)
            else:
                target = self.target_type(target)
        return feature, target

    def transform_typesignature(self, feature_typesig, target_typesig=None):
        if self.feature_type is None or feature_typesig is None:
            new_feature_typesig = feature_typesig
        else:
            new_feature_typesig = TypeSignature(
                datatype=self.feature_type,
                datarange=feature_typesig.datarange.astype(self.feature_type),
            )
        if self.target_type is None or target_typesig is None:
            new_target_typesig = target_typesig
        else:
            new_target_typesig = TypeSignature(
                datatype=self.target_type,
                datarange=target_typesig.datarange.astype(self.target_type),
            )
        return new_feature_typesig, new_target_typesig


class ResampleTransform(Transform):
    params = ["feature_up", "target_up", "zoom_kwargs"]

    def _call(self, feature, target=None):
        zoom_kwargs = self.zoom_kwargs or dict()
        if self.feature_up is not None:
            if len(self.feature_up) != len(feature.shape):
                raise ValueError(
                    "Wrong length for feature_up (%i) "
                    "given feature.shape=%s" % (len(self.feature_up), feature.shape)
                )
            feature = zoom(feature, zoom=self.feature_up, **zoom_kwargs)
        if target is not None and self.target_up is not None:
            if len(self.target_up) != len(target.shape):
                raise ValueError(
                    "Wrong length for target_up (%i) "
                    "given target.shape=%s" % (len(self.target_up), target.shape)
                )
            target = zoom(target, zoom=self.target_up, **zoom_kwargs)
        return feature, target


class ClipFeatureTransform(Transform):
    params = ["lower", "upper"]

    def _call(self, feature, target=None):
        feature = np.clip(feature, a_min=self.lower, a_max=self.upper)
        return feature, target

    def transform_typesignature(self, feature_typesig, target_typesig=None):
        if feature_typesig is None:
            return None, target_typesig
        new_feature_typesig = TypeSignature(
            datatype=feature_typesig.datatype,
            datarange=np.array(
                (self.lower, self.upper), dtype=feature_typesig.datatype
            ),
            rangetype="minmax",
        )
        return new_feature_typesig, target_typesig


class RescaleFeatureTransform(Transform):
    params = dict(shift=0.0, scale=1.0)

    def _call(self, feature, target=None):
        feature -= self.shift
        feature /= self.scale
        return feature, target

    def transform_typesignature(self, feature_typesig, target_typesig=None):
        if feature_typesig is None:
            return None, target_typesig
        if feature_typesig.rangetype == "minmax":
            new_datarange = feature_typesig.datarange
            new_datarange -= self.shift
            new_datarange /= self.scale
            # If self.scale is negative, the min/max of the range may be swapped
            new_datarange = np.array((new_datarange.min(), new_datarange.max()))
        elif feature_typesig.rangetype == "stddev":
            mean = np.mean(feature_typesig.datarange)
            std = feature_typesig.datarange[1] - mean
            new_mean = (mean - self.shift) / self.scale
            new_std = std / np.abs(self.scale)
            new_datarange = np.array((new_mean - new_std, new_mean + new_std))
        else:
            raise ValueError()
        new_feature_typesig = TypeSignature(
            datatype=feature_typesig.datatype,
            datarange=new_datarange,
            rangetype=feature_typesig.rangetype,
        )
        return new_feature_typesig, target_typesig


class SquareImageTransform(Transform):
    """Return square feature and target from rectangular feature or target"""

    def _call(self, feature, target=None):
        h, w, n = feature.shape
        w = min(w, h)
        feature = feature[:w, :w]  # force square image
        if target is not None:
            h, w = target.shape[:2]
            w = min(w, h)
            target = target[:w, :w]  # force square image
        return feature, target


class FlipFeatureMultioutTransform(Transform):
    """Flip the feature according to the symmetries of the square."""

    multiplier = 8

    def _call(self, feature, target=None):
        return symmetries_of_the_square(feature), [target] * 8


class FlipFeatureTargetMultioutTransform(Transform):
    """Flip the feature and target according to symmetries of the square."""

    multiplier = 8

    def _call(self, feature, target=None):
        if target is None:
            return symmetries_of_the_square(feature), target
        else:
            return (symmetries_of_the_square(feature), symmetries_of_the_square(target))


class FlipFeatureTargetTransform(Transform):
    """Return one flipped orientation of feature and target
       according to the symmetries of the square."""

    def _call(self, feature, target=None):
        flip = np.random.randint(8)
        feature = symmetries_of_the_square(feature, flip_indices=[flip])
        if target is None:
            return feature, target
        return (feature, symmetries_of_the_square(target, flip_indices=[flip]))


class FlipFeatureTransform(Transform):
    """Return one flipped orientation of feature and target
       according to the symmetries of the square."""

    def _call(self, feature, target=None):
        flip = np.random.randint(8)
        feature = symmetries_of_the_square(feature, flip_indices=[flip])
        return feature, target


class AdditiveNoiseTransform(Transform):
    """
    Add a random offset to all bands in feature.
    The random offset is centered on 0 with a std of ADDITIVE_NOISE
    """

    params = ["additive_noise", "verbose"]

    def _call(self, feature, target=None):
        if self.additive_noise:
            add = np.random.randn() * self.additive_noise
            feature += add
            if self.verbose:
                print("Additive noise: %f" % add)
        return feature, target


class MultiplicativeNoiseTransform(Transform):
    """
    Multiply all bands in feature by a positive random scaling factor.
    The random scaling factor is drawn from a distribution centered on 1 with
    a width dictated by MULTIPLICATIVE_NOISE.
    """

    params = dict(multiplicative_noise=None, verbose=False)

    def _call(self, feature, target=None):
        if self.multiplicative_noise:
            mult = np.exp(np.random.randn() * self.multiplicative_noise)
            feature *= mult
            if self.verbose:
                print("Multiplicative noise: %f" % mult)
        return feature, target


#########
# bbox-specific transforms


def _rotate_bbox(target, h, w, theta):
    cx = w // 2
    cy = h // 2
    m = np.array(
        [
            [
                np.cos(np.deg2rad(theta)),
                np.sin(np.deg2rad(theta)),
                (1 - (np.cos(np.deg2rad(theta)))) * cx - np.sin(np.deg2rad(theta)) * cy,
            ],
            [
                -1 * np.sin(np.deg2rad(theta)),
                np.cos(np.deg2rad(theta)),
                np.sin(np.deg2rad(theta)) * cx + (1 - (np.cos(np.deg2rad(theta)))) * cy,
            ],
        ]
    )
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    m[0, 2] += (nw / 2) - cx
    m[1, 2] += (nh / 2) - cy

    new_target = []
    for t in target:
        bb = [(t[1], t[3]), (t[2], t[3]), (t[2], t[4]), (t[1], t[4])]
        new_bb = list(bb)
        for i, coord in enumerate(bb):
            v = [coord[0], coord[1], 1]
            calculated = np.dot(m, v)
            new_bb[i] = (calculated[0], calculated[1])
        new_bb = np.array(new_bb)
        new_t = np.array(
            (
                t[0],
                int(np.min(new_bb[:, 0])),
                int(np.max(new_bb[:, 0])),
                int(np.min(new_bb[:, 1])),
                int(np.max(new_bb[:, 1])),
            )
        )
        new_target.append(new_t)
    return np.array(new_target)


class BBoxFlipUDFeatureTargetTransform(Transform):
    """Vertically flips orientation of image feature and bbox target"""

    def _call(self, feature, target=None):
        if np.random.random() > 0.5:
            feature = np.flipud(feature)
            if target is not None:
                target[:, (3, 4)] = feature.shape[0] - target[:, (4, 3)]
        return feature, target


class BBoxRotate90FeatureTargetTransform(Transform):
    """Return one of 4 rotations of 90deg of image feature and bbox target"""

    def _call(self, feature, target=None):
        krot = np.random.randint(0, 4)
        feature = np.rot90(feature, k=krot)
        if target is not None:
            target = _rotate_bbox(target, feature.shape[0], feature.shape[1], krot * 90)
        return feature, target


#########
# fit parameters-specific transforms
class ColorJitterWithFitParamsTransform(Transform):
    params = dict(mean=0.0, pca=1.0, probability=0.5, magnitude=0.1)

    def _call(self, feature, target=None):
        if np.random.random() <= self.probability:
            perturb = (self.pca * np.random.randn(len(self.mean)) * self.magnitude).sum(
                axis=1
            )
            feature += perturb
            return feature, target
        return feature, target


#########
# SSD transforms
class ApplySSDBoxEncoderTransform(Transform):
    """Encode bounding box target for the loss function.

    An encoder can be passed in, or an encoder will be re-created from provided metadata.
    To check the list of model parameters (both, required and optional),
    look under `load_ssd300_encoder` of keras/ssd300_utils.py (and further SSDBoxEncoder if required)

    Parameters
    ----------
    encoder : object of type appsci_utils.projects.ssd300.ssd_box_encode_decode_utils.SSDBoxEncoder
        Encoder object
    """

    params = ["encoder", "meta"]

    def _call(self, feature, target=None):
        from appsci_utils.models.keras.ssd300_utils import (
            SSDBoxEncoder,
            load_ssd300_encoder,
        )

        # If encoder is passed in, ensure it is the right type
        if self.encoder:
            if not isinstance(self.encoder, SSDBoxEncoder):
                raise ValueError(
                    "Encoder must be an object of type appsci_utils.keras.ssd300_utils."
                    "SSDBoxEncoder. Found: {}".format(type(self.encoder))
                )

        # If encoder was not passed in, load the encoder from metadata
        else:
            # Check metadata
            if hasattr(self.meta, "kwargs_for_model"):
                self.meta = self.meta["kwargs_for_model"]
            try:
                self.encoder = load_ssd300_encoder(meta=self.meta)
            except Exception as error:
                raise Exception("Failed to load SSD300 encoder") from error

        # Encode target using this encoder
        target = self.encoder.encode_y([target])
        return feature, target.squeeze()


# Autoencoder utilities
class DummyTargetTransform(Transform):
    params = ["target_shapes"]

    def _call(self, feature, target=None):
        return feature, [np.empty(shape) for shape in self.target_shapes]


# Other
class OneHotTransform(Transform):
    """Turn target integer labels into a one-hot representation. Label dimension last."""

    params = ["n_labels"]

    def _call(self, feature, target=None):
        if target is None:
            return feature, target

        if not isinstance(target, np.ndarray):
            raise ValueError("OneHotTransform expects ndarrays")

        new_target = np.zeros(target.shape + (self.n_labels,), dtype=np.float32)
        new_target[..., target] = 1
        return feature, new_target

    def transform_typesignature(self, feature_typesig, target_typesig=None):
        return feature_typesig, TypeSignature(datatype=np.float32, datarange=(0, 1))
