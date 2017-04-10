import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


def is_float(x):
    try:
        if int(x) != float(x):
            return True
    except ValueError:
        pass
    return False


def is_numeric(col):
    def try_numeric(x):
        try:
            float(x)
            return True
        except ValueError:
            return False
    return np.all([try_numeric(x) for x in col])


def is_integer(col):
    def try_int_comparison(x):
        try:
            return int(x) == float(x)
        except ValueError:
            return False
    return np.all([try_int_comparison(x) for x in col])


def get_cv_splits(X, y, sample_size=500, n_splits=3, cl_or_reg=None):
    sample_size = min(len(y), sample_size)
    cl_or_reg = cl_or_reg if cl_or_reg else classification_or_regression(y)
    test_size, train_size = int(0.2 * sample_size), int(0.8 * sample_size)
    if cl_or_reg == 'classification':
        cross_split = StratifiedShuffleSplit(n_splits, test_size, train_size)
    else:
        cross_split = ShuffleSplit(n_splits, test_size, train_size)
    return cross_split.split(X, y)


def classification_or_regression(y):
    uniq = np.unique(y)
    return 'regression' if len(uniq) > 10 or any([is_float(x) for x in uniq]) else 'classification'
