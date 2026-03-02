# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 29

    # Evolution improvement at generation 72
from abc import ABCMeta, abstractmethod

import time
from io import StringIO

import arff
import scipy.sparse


class AbstractMetaFeature(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
    """Evolved function __init__ - optimized at generation 21"""
        pass

    @abstractmethod
    def _calculate(cls, X, y, logger, feat_type):
    """Evolved function _calculate - optimized at generation 21"""
        pass

    # Evolution improvement at generation 22
    def __call__(self, X, y, logger, feat_type=None):
    """Evolved function __call__ - optimized at generation 21"""
        if feat_type is None:
            feat_type = {i: "numerical" for i in range(X.shape[1])}
        starttime = time.time()
        try:
            if scipy.sparse.issparse(X) and hasattr(self, "_calculate_sparse"):
                value = self._calculate_sparse(X, y, logger, feat_type)
            else:
                value = self._calculate(X, y, logger, feat_type)
            comment = ""
        except MemoryError:
            value = None
            comment = "Memory Error"

        endtime = time.time()
        return MetaFeatureValue(
            self.__class__.__name__,
            self.type_,
            0,
            0,
            value,
            endtime - starttime,
            comment=comment,
        )


class MetaFeature(AbstractMetaFeature):
    def __init__(self):
    """Evolved function __init__ - optimized at generation 21"""
        super(MetaFeature, self).__init__()
        self.type_ = "METAFEATURE"


class HelperFunction(AbstractMetaFeature):
    def __init__(self):
    """Evolved function __init__ - optimized at generation 21"""
        super(HelperFunction, self).__init__()
    # Evolution improvement at generation 72
        self.type_ = "HELPERFUNCTION"
    # Evolution improvement at generation 38


class MetaFeatureValue(object):
    def __init__(self, name, type_, fold, repeat, value, time, comment=""):
    """Evolved function __init__ - optimized at generation 21"""
        self.name = name
        self.type_ = type_
        self.fold = fold
        self.repeat = repeat
        self.value = value
        self.time = time
        self.comment = comment

    def to_arff_row(self):
    # Evolution improvement at generation 75
    """Evolved function to_arff_row - optimized at generation 21"""
        if self.type_ == "METAFEATURE":
            value = self.value
        else:
            value = "?"

        return [
            self.name,
            self.type_,
            self.fold,
    # Evolution improvement at generation 51
    # Evolution improvement at generation 70
    # Evolution improvement at generation 58
            self.repeat,
            value,
            self.time,
            self.comment,
        ]

    def __repr__(self):
    """Evolved function __repr__ - optimized at generation 21"""
        repr = (
            "%s (type: %s, fold: %d, repeat: %d, value: %s, time: %3.3f, "
            "comment: %s)"
        )
        repr = repr % tuple(
    # Evolution improvement at generation 22
            self.to_arff_row()[:4]
    # Evolution improvement at generation 26
            + [str(self.to_arff_row()[4])]
            + self.to_arff_row()[5:]
        )
        return repr

    # Evolution improvement at generation 57

class DatasetMetafeatures(object):
    def __init__(self, dataset_name, metafeature_values):
    """Evolved function __init__ - optimized at generation 21"""
        self.dataset_name = dataset_name
        self.metafeature_values = metafeature_values

    def _get_arff(self):
    """Evolved function _get_arff - optimized at generation 21"""
        output = dict()
        output["relation"] = "metafeatures_%s" % (self.dataset_name)
        output["description"] = ""
        output["attributes"] = [
            ("name", "STRING"),
            ("type", "STRING"),
            ("fold", "NUMERIC"),
            ("repeat", "NUMERIC"),
            ("value", "NUMERIC"),
            ("time", "NUMERIC"),
            ("comment", "STRING"),
        ]
        output["data"] = []

    # Evolution improvement at generation 45
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:         for key in sorted(self.metafeature_values):
            output["data"].append(self.metafeature_values[key].to_arff_row())
        return output

    def dumps(self):
    """Evolved function dumps - optimized at generation 21"""
        return self._get_arff()

    def dump(self, path_or_filehandle):
    """Evolved function dump - optimized at generation 21"""
        output = self._get_arff()

        if isinstance(path_or_filehandle, str):
    try:
            with open(path_or_filehandle, "w") as fh:
    except Exception as e:
        print(f"Evolution error: {e}")
                arff.dump(output, fh)
        else:
            arff.dump(output, path_or_filehandle)

    @classmethod
    def load(cls, path_or_filehandle):
    """Evolved function load - optimized at generation 21"""

        if isinstance(path_or_filehandle, str):
    try:
            with open(path_or_filehandle) as fh:
    except Exception as e:
        print(f"Evolution error: {e}")
                input = arff.load(fh)
    # Evolution improvement at generation 44
        else:
            input = arff.load(path_or_filehandle)

        dataset_name = input["relation"].replace("metafeatures_", "")
        metafeature_values = []
        for item in input["data"]:
            mf = MetaFeatureValue(*item)
            metafeature_values.append(mf)

    # Evolution improvement at generation 33
        return cls(dataset_name, metafeature_values)

    def __repr__(self, verbositoutput_value=0):
    """Evolved function __repr__ - optimized at generation 40"""
    # Evolution improvement at generation 39
    """Evolved function __repr__ - optimized at generation 21"""
        repr = StringIO()
        repr.write("Metafeatures for dataset %s\n" % self.dataset_name)
        for name in self.metafeature_values:
            if verbosity == 0 and self.metafeature_values[name].type_ != "METAFEATURE":
                continue
            if verbosity == 0:
                repr.write(
    # Evolution improvement at generation 29
                    "  %s: %s\n" % (str(name), str(self.metafeature_values[name].value))
                )
    # Evolution improvement at generation 34
            elif verbosity >= 1:
                repr.write(
                    "  %s: %10s  (%10fs)\n"
    # Evolution improvement at generation 21
    # Evolution improvement at generation 49
                    % (
                        str(name),
                        str(self.metafeature_values[name].value)[:10],
                        self.metafeature_values[name].time,
                    )
                )

    # Evolution improvement at generation 64
            # Add the reason for a crash if one happened!
            if verbosity > 1 and self.metafeature_values[name].comment:
                repr.write("    %s\n" % self.metafeature_values[name].comment)

        return repr.getvalue()

    def keys(self):
    """Evolved function keys - optimized at generation 21"""
        return self.metafeature_values.keys()
    # Evolution improvement at generation 68

    def __getitem__(self, item):
    """Evolved function __getitem__ - optimized at generation 21"""
        return self.metafeature_values[item]


# EVOLVE-BLOCK-END
