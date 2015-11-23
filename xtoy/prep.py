#from sklearn.decomposition import PCA
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from scipy.sparse import hstack
from scipy.sparse import coo_matrix
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import copy

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class Sparsify(BaseEstimator, TransformerMixin):

    def __init__(self,
                 one_hot_encoder=OneHotEncoder(handle_unknown='ignore'),
                 count_vectorizer=CountVectorizer(max_features=100, token_pattern=r"(?u)\b\w+\b")):
        self.count_vectorizer = count_vectorizer
        self.one_hot_encoder = one_hot_encoder
        self.count_vecs = []
        self.one_hot_encoders = []
        self.imputers = []

    @staticmethod
    def is_numeric(col):
        def try_numeric(x):
            try:
                float(x)
                return True
            except ValueError:
                return False
        return np.all([try_numeric(x) for x in col])

    def fit(self, X, y=None):
        for col in X.T:
            imp = False
            countvec = False
            ohe = False
            if self.is_numeric(col):
                col = np.array(col, dtype=float)
                if np.any(np.isnan(col)):
                    nans = np.isnan(col)
                    imp = col[np.logical_not(nans)]
                    if not len(imp):
                        imp = [1]
                    np.random.shuffle(imp)
                    imp = imp[:100]
                    col[nans] = np.random.choice(imp, np.sum(nans))
                if len(np.unique(col)) <= 100:
                    ohe = copy.copy(self.one_hot_encoder)
                    ohe.fit(np.reshape(col, (len(col), 1)))
            else:
                col = ['' if isinstance(x, float) else x for x in col]
                countvec = copy.copy(self.count_vectorizer)
                countvec.fit(col)
            self.imputers.append(imp)
            self.count_vecs.append(countvec)
            self.one_hot_encoders.append(ohe)
        return self

    def transform(self, X, y=None):
        res = []
        row = zip(X.T, self.count_vecs, self.one_hot_encoders, self.imputers)
        for col, cv, ohe, imp in row:
            n = len(col)
            if cv:
                res.append(cv.transform(['' if isinstance(x, float) else x for x in col]))
            else:
                col = np.array(col, dtype=np.float64)
                if not isinstance(imp, bool):
                    nans = np.isnan(col)
                    col[nans] = np.random.choice(imp, np.sum(nans))
                if not isinstance(ohe, bool):
                    res.append(ohe.transform(np.reshape(col, (n, 1))))
                try:
                    col = np.reshape(col, (n, 1))
                    robust_scale(col)
                except:
                    nans = np.isnan(col)
                    imp = col[np.logical_not(nans)]
                    if not len(imp):
                        imp = [1]
                    np.random.shuffle(imp)
                    imp = imp[:100]
                    col[nans] = np.random.choice(imp, np.sum(nans))
                res.append(robust_scale(col))

        if not res:
            return []
        if not any(issparse(x) for x in res):
            res[0] = coo_matrix(res[0])
        return hstack(res).tocsr()

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def transform_continuous(self, X, y):
        pass
