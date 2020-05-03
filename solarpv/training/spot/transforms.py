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


class FlipFeatureTargetTransform(Transform):
    """Return one flipped orientation of feature and target
       according to the symmetries of the square."""

    def _call(self, feature, target=None):
        flip = np.random.randint(8)
        feature = symmetries_of_the_square(feature, flip_indices=[flip])
        if target is None:
            return feature, target
        return (feature, symmetries_of_the_square(target, flip_indices=[flip]))


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
