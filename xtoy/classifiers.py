#from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from xtoy.utils import is_float

#ridge_grid = {'clf__alpha': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.4, 1.7, 10, 100, 1000]}

ridge_grid = {'clf__alpha': [0.1, 1., 10.], 'clf__normalize': [True]}

ridge_classification = {'clf': RidgeClassifier, 'grid': ridge_grid}
ridge_regression = {'clf': Ridge, 'grid': ridge_grid}
# ridge_regression = {'clf': svm.SVR, 'grid': {
#     'clf__shrinking': [True, False],
#     'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#     'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}}


rf_grid = {'clf__max_features': ['sqrt', 'auto', 'log2', 0.5, 0.8, 0.9],
           'clf__max_depth': [None, 5, 10, 20],
           'clf__min_samples_leaf': [1, 5, 10],
           'clf__min_samples_split': [1, 10, 20],
           'clf__n_estimators': [200]}

rf_grid_classification = rf_grid.copy()
rf_grid_classification.update({'clf__class_weight': ['balanced']})

rf_classification = {'clf': RandomForestClassifier, 'grid': rf_grid_classification}
rf_regression = {'clf': RandomForestRegressor, 'grid': rf_grid}


options = {
    'regression': {
        'dense': [ridge_regression, rf_regression],
        'sparse': [ridge_regression]
    },
    'classification': {
        'dense': [ridge_classification, rf_classification],
        'sparse': [ridge_classification]
    }
}


def sparse_or_dense(X, RAM=None, magic=42):
    # USING PAPER WE WILL GO TO DENSE ANYWAY
    return 'dense'
    # size = np.prod(X.shape) if hasattr(X, 'shape') else len(X) * len(X[0])

    # # if N * num_feat * magic > RAM:
    # if size > 1000 ** 3:
    #     return 'sparse'
    # else:
    #     return 'dense'


def classification_or_regression(y):
    uniq = np.unique(y)
    return 'regression' if len(uniq) > 10 or any(map(is_float, uniq)) else 'classification'


def pick(X, y, cl_or_reg=None, opts=options):
    # i always choose first one now
    cl_or_reg = cl_or_reg if cl_or_reg else classification_or_regression(y)
    chosen_model = dict(opts[cl_or_reg][sparse_or_dense(X)][0])
    # chosen_model['grid'].update({'tsvd__n_iter': [5, 10, 20],
    #                             'tsvd__n_components': [10, 20]})
    return chosen_model
