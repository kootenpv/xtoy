# from sklearn.svm import SVC
import scipy
from sklearn import svm

# from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from xtoy.utils import classification_or_regression


ridge_grid = {
    "estimator__alpha": [
        0.00001,
        0.001,
        0.01,
        0.05,
        0.1,
        0.3,
        0.5,
        0.8,
        1.0,
        1.4,
        1.7,
        10,
        100,
        1000,
        10000,
        100000,
    ]
}
# normalize seems bugged at prediction time!

# ridge_grid = {'estimator__alpha': [0.1, 1., 10.]}

INF = float("inf")
ridge_classification = {
    "clf": RidgeClassifier,
    "grid": ridge_grid,
    "name": "ridge",
    "max_complexity": INF,
}
ridge_regression = {"clf": Ridge, "grid": ridge_grid, "name": "ridge", "max_complexity": INF}

# ridge_regression = {'clf': svm.SVR, 'grid': {
#     'estimator__shrinking': [True, False],
#     'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#     'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}}


rf_grid = {
    "estimator__max_features": ["sqrt", "auto", "log2", 0.5, 0.8, 0.9],
    "estimator__max_depth": [None, 5, 10, 20],
    "estimator__min_samples_leaf": [1, 5, 10],
    "estimator__min_samples_split": [2, 10, 20],
    "estimator__n_estimators": [1000],
}

knn_grid = {
    "estimator__n_neighbors": [1, 2, 3, 5, 10, 20],
    "estimator__leaf_size": [2, 3, 5, 10, 30, 50, 100],
    "estimator__p": [1, 2, 5, 10],
    "estimator__weights": ["uniform", "distance"],
}

rf_grid_classification = rf_grid.copy()
rf_grid_classification.update({"estimator__class_weight": ["balanced"]})

rf_classification = {
    "clf": RandomForestClassifier,
    "grid": rf_grid_classification,
    "name": "rf",
    "max_complexity": 100000 * 10,
}
rf_regression = {
    "clf": RandomForestRegressor,
    "grid": rf_grid,
    "name": "rf",
    "max_complexity": 100000 * 10,
}

knn_classification = {
    "clf": KNeighborsClassifier,
    "grid": knn_grid,
    "name": "knn",
    "max_complexity": 100000 * 100,
}
knn_regression = {
    "clf": KNeighborsRegressor,
    "grid": knn_grid,
    "name": "knn",
    "max_complexity": 100000 * 100,
}


options = {
    "regression": [ridge_regression, rf_regression, knn_regression],
    "classification": [ridge_classification, rf_classification, knn_classification],
}

xgb_grid = {
    "estimator__max_depth": [2, 3, 5, 10, 20],
    "estimator__min_child_weight": [1, 3, 5, 10, 15, 20],
    "estimator__learning_rate": [0.0001, 0.001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1],
    "estimator__n_estimators": [100],
}


try:
    import xgboost

    xgb_classification = {
        "clf": xgboost.XGBClassifier,
        "grid": xgb_grid,
        "name": "xgb",
        "max_complexity": 100000 * 10,
    }
    xgb_regression = {
        "clf": xgboost.XGBRegressor,
        "grid": xgb_grid,
        "name": "xgb",
        "max_complexity": 100000 * 10,
    }
    options["classification"].append(xgb_classification)
    options["regression"].append(xgb_regression)
except ImportError:
    pass

lgb_grid = {
    "estimator__num_leaves": [2, 3, 5, 10, 20, 31, 50],
    "estimator__min_child_weight": [1, 3, 5, 10, 15, 20],
    "estimator__learning_rate": [0.0001, 0.001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1],
}

try:
    import lightgbm as lgb

    lgb_classification = {
        "clf": lgb.LGBMClassifier,
        "grid": lgb_grid,
        "name": "lgb",
        "max_complexity": 100000 * 10,
    }
    lgb_regression = {
        "clf": lgb.LGBMRegressor,
        "grid": lgb_grid,
        "name": "lgb",
        "max_complexity": 100000 * 10,
    }
    options["classification"].append(lgb_classification)
    options["regression"].append(lgb_regression)
except ImportError:
    pass


def sparse_or_dense(X, RAM=None, magic=42):
    if scipy.sparse.issparse(X):
        return "sparse"
    else:
        # USING PAPER WE WILL GO TO DENSE ANYWAY
        return "dense"
    # size = np.prod(X.shape) if hasattr(X, 'shape') else len(X) * len(X[0])

    # # if N * num_feat * magic > RAM:
    # if size > 1000 ** 3:
    #     return 'sparse'
    # else:
    #     return 'dense'


def pick(X, y, cl_or_reg=None, opts=None):
    # i always choose first one now
    if opts is None:
        opts = options
    cl_or_reg = cl_or_reg if cl_or_reg else classification_or_regression(y)
    return opts[cl_or_reg]
