# PROBEER OOIT NOG EEN KEER OM DE countvec OP MAX TE ZETTEN OFZO
# https://github.com/paulgb/sklearn-pandas

# from sklearn.decomposition import PCA
import copy
import re
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                   robust_scale)
from dateutil.parser import parse
from xtoy.utils import is_numeric
from xtoy.utils import is_integer


def merge_dicts(dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


class RegexVectorizer(TransformerMixin):

    def __init__(self, regex_patterns=None, ignore_case=True, dict_vectorizer=DictVectorizer(),
                 binary=False):
        self.ignore_case = ignore_case
        self.binary = binary
        flags = ignore_case * re.IGNORECASE
        self.regex_patterns = regex_patterns
        self.regexes = {k: re.compile(v, flags=flags) for k, v in regex_patterns.items()}
        self.dict_vectorizer = dict_vectorizer

    def _featurize(self, X):
        if self.binary:
            feats = [{name: 1 for name, r in self.regexes.items() if r.search(x)} for x in X]
        else:
            feats = [merge_dicts([Counter(name + "_" + m for m in r.findall(x) + ["total"])
                                  for name, r in self.regexes.items()])
                     for x in X]
        return feats

    def fit(self, X, y=None):
        feats = self._featurize(X)
        self.dict_vectorizer.fit(feats)
        return self

    def transform(self, X, y=None):
        feats = self._featurize(X)
        return self.dict_vectorizer.transform(feats)


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
        self.fill = None

    @staticmethod
    def most_frequent(col):
        try:
            return col.value_counts().index[0]
        except IndexError:
            return 0

    def get_impute_val(self, col):
        if col.dtype == np.dtype('O'):
            val = self.most_frequent(col)
        elif col.dtype == np.dtype(float):
            val = col.mean()
        else:
            val = col.median()
        if isinstance(val, float) and np.isnan(val):
            val = 0
        return val

    def fit(self, X, y=None):
        self.fill = pd.Series([self.get_impute_val(X[c]) for c in X], index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill, inplace=False)


class Sparsify(BaseEstimator, TransformerMixin):

    def __init__(self,
                 count_vectorizer=CountVectorizer(max_features=100, token_pattern=r"(?u)\b\w+\b"),
                 max_unique_for_discrete=15,
                 date_atts=("year", "month", "day", "weekday", "hour", "minute", "second")):
        self.count_vectorizer = count_vectorizer
        self.max_unique_for_discrete = max_unique_for_discrete
        self.one_hot_encoder = None
        # scary duplication
        self.drop_vars = []
        self.count_vecs = []
        self.date_vars = []
        self.imputer = DataFrameImputer()
        self.ohe_indices = []
        self.numeric_indices = []
        self.var_names_ = None
        self.date_atts = date_atts

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = self.imputer.fit_transform(X)
        self.ohe_indices = np.zeros(X.shape[1], dtype=bool)
        self.numeric_indices = np.zeros(X.shape[1], dtype=bool)
        self.drop_vars = np.zeros(X.shape[1], dtype=bool)
        self.date_vars = np.zeros(X.shape[1], dtype=bool)
        self.var_names_ = []
        for i, col in enumerate(X):
            if len(set(X[col])) == 1:
                self.drop_vars[i] = True

        for i, col in enumerate(X):
            if self.drop_vars[i]:
                continue
            if is_numeric(X[col]):
                if len(set(X[col])) == 2:
                    self.var_names_.append('{}_{}_continuous'.format(col, i))
                else:
                    self.var_names_.append('{}_{}_dummy'.format(col, i))
                num_unique = len(np.unique(X[col]))
                if is_integer(X[col]) and 3 <= num_unique <= self.max_unique_for_discrete:
                    self.ohe_indices[i] = True
                self.numeric_indices[i] = True
            # maybe date
            else:
                for x in X[col]:
                    try:
                        parse(x)
                    except ValueError:
                        break
                else:
                    self.date_vars[i] = True
                    date_var_names = ['{}_date_{}'.format(col, x)
                                      for j, x in enumerate(self.date_atts)]
                    self.var_names_.extend(date_var_names)

        for i, (ohed, col) in enumerate(zip(self.ohe_indices, X)):
            if not self.drop_vars[i] and not is_numeric(X[col]) and not ohed and not self.date_vars[i]:
                countvec = copy.copy(self.count_vectorizer)
                countvec.fit(['' if isinstance(x, float) else x for x in X[col]])
                self.count_vecs.append(countvec)
                cv_var_names = ['{}_countvec_{}_{}_{}'.format(col, x, i, j)
                                for j, x in enumerate(countvec.get_feature_names())]
                self.var_names_.extend(cv_var_names)
            else:
                self.count_vecs.append(False)

        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
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
        date_groups = []
        for i, col in enumerate(X):
            if self.drop_vars[i] or self.ohe_indices[i] or self.count_vecs[i]:
                continue
            if self.numeric_indices[i]:
                res.append(X[col])
            elif self.date_vars[i]:
                prefix = str(col) + "__"
                dtimes = [parse(x) for x in X[col]]
                for a in self.date_atts:
                    dcol = [x.weekday() if a == "weekday" else getattr(x, a) for x in dtimes]
                    res.append(dcol)
        for cv, col in zip(self.count_vecs, X):
            if cv:
                cvs.append(cv.transform(['' if isinstance(x, (np.int64, int, float)) else x
                                         for x in X[col]]))
        combined = [scipy.sparse.coo_matrix(res).T] + cvs
        if np.any(self.ohe_indices):
            ohd = self.one_hot_encoder.transform(X[X.columns[self.ohe_indices]])
            combined = combined + [ohd]

        return scipy.sparse.hstack(combined).tocsr()
