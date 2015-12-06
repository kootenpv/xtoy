# PROBEER OOIT NOG EEN KEER OM DE countvec OP MAX TE ZETTEN OFZO

#from sklearn.decomposition import PCA
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import copy

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):
    """
    Credits http://stackoverflow.com/a/25562948/1575066
    """

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):

        self.fill = pd.Series([
            X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else
            X[c].mean() if X[c].dtype == np.dtype(float) else X[c].median()
            for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill, inplace=False)


class Sparsify(BaseEstimator, TransformerMixin):

    def __init__(self,
                 count_vectorizer=CountVectorizer(max_features=100, token_pattern=r"(?u)\b\w+\b")):
        self.count_vectorizer = count_vectorizer
        self.one_hot_encoder = None
        self.count_vecs = []
        self.one_hot_encoder = []
        self.imputer = DataFrameImputer()
        self.ohe_indices = []
        self.var_names_ = None

    @staticmethod
    def is_numeric(col):
        def try_numeric(x):
            try:
                float(x)
                return True
            except ValueError:
                return False
        return np.all([try_numeric(x) for x in col])

    @staticmethod
    def is_integer(col):
        def try_int_comparison(x):
            try:
                return int(x) == float(x)
            except ValueError:
                return False
        return np.all([try_int_comparison(x) for x in col])

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = self.imputer.fit_transform(X)
        self.ohe_indices = np.zeros(X.shape[1], dtype=bool)
        self.var_names_ = []
        for i, col in enumerate(X):
            if self.is_numeric(X[col]):
                self.var_names_.append('{}_{}_continuous'.format(col, i))
                if self.is_integer(X[col]) and len(np.unique(X[col])) <= 100:
                    self.ohe_indices[i] = True

        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        for i, (h, col) in enumerate(zip(self.ohe_indices, X)):
            if not self.is_numeric(X[col]) and not h:
                countvec = copy.copy(self.count_vectorizer)
                countvec.fit(['' if isinstance(x, float) else x for x in X[col]])
                self.count_vecs.append(countvec)
                cv_var_names = ['{}_countvec_{}_{}_{}'.format(col, x, i, j)
                                for j, x in enumerate(countvec.get_feature_names())]
                self.var_names_.extend(cv_var_names)
            else:
                self.count_vecs.append(False)
        if np.any(self.ohe_indices):
            self.one_hot_encoder.fit(X[X.columns[self.ohe_indices]])
            # ridiculously complex OHE feature range names
            feat = set(self.one_hot_encoder.active_features_)
            for l, h, v in zip(self.one_hot_encoder.feature_indices_[:-1],
                               self.one_hot_encoder.feature_indices_[1:],
                               X.columns[self.ohe_indices]):
                self.var_names_.extend(['{}_OHE_{}'.format(v, j)
                                        for j in range(len(feat.intersection(range(l, h))))])
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = self.imputer.transform(X)
        res = []
        cvs = []
        for col in X:
            if self.is_numeric(X[col]):
                res.append(X[col])
        for cv, col in zip(self.count_vecs, X):
            if cv:
                cvs.append(cv.transform(['' if isinstance(x, float) else x for x in X[col]]))
        if np.any(self.ohe_indices):
            ohd = self.one_hot_encoder.transform(X[X.columns[self.ohe_indices]])
            combined = [scipy.sparse.coo_matrix(res).T] + cvs + [ohd]
        else:
            combined = [scipy.sparse.coo_matrix(res).T] + cvs
        return scipy.sparse.hstack(combined).tocsr()
