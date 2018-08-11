import warnings
import re
from collections import Counter

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer


class RegexVectorizer(TransformerMixin):

    def __init__(self, regex_patterns=None, ignore_case=False, dict_vectorizer=DictVectorizer(),
                 binary=False):
        self.ignore_case = ignore_case
        self.binary = binary
        flags = ignore_case * re.IGNORECASE
        self.regex_patterns = regex_patterns
        self.regexes = {k: re.compile(v, flags=flags) for k, v in regex_patterns.items()}
        self.dict_vectorizer = dict_vectorizer

    def get_feature_names(self):
        return self.dict_vectorizer.get_feature_names()

    def fit(self, X, y=None):
        feats = self.featurize(X)
        self.dict_vectorizer.fit(feats)
        return self

    def transform(self, X, y=None):
        feats = self.featurize(X)
        return self.dict_vectorizer.transform(feats)

    @staticmethod
    def _merge_dicts(dict_args):
        result = Counter()
        for dictionary in dict_args:
            for k, v in dictionary.items():
                result[k] += v
        return result

    def featurize(self, X):
        if self.binary:
            feats = [{name: 1 for name, r in self.regexes.items() if r.search(x)} for x in X]
        else:
            feats = [self._merge_dicts([self._get_dict(name, r, x)
                                        for name, r in self.regexes.items()])
                     for x in X]
        return feats

    @staticmethod
    def _get_dict(name, r, text_value):
        matches = r.findall(text_value)
        if not matches:
            return Counter()
        return Counter(name + "_" + m for m in matches + ["_any"])

# X = ["I walked", "I am walking"]
# rv = RegexVectorizer({"walk": "walk[^ ]"})
# rv.fit_transform(X).todense()
# rv.featurize(X)


class SelectPolynomialByResidual():
    def __init__(self, clf, n_to_select="auto", test_size=0.4, feature_indices=None):
        self.clf = clf
        self.test_size = test_size
        self.feature_names = None
        self.feature_indices = None if feature_indices is None else feature_indices
        self.n_to_select = n_to_select
        self.best_coefs = None

    def transform(self, X, y=None):
        poly_features = []
        feature_indices = set(self.feature_indices)
        it = 0
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                if i <= j:
                    if it in feature_indices:
                        poly_features.append(X[:, i] * X[:, j])
                    it += 1
        poly_features = np.vstack(poly_features).T
        return np.hstack((X, poly_features))

    def fit(self, X, y, var_names):
        # hack
        if self.feature_indices is not None:
            return self
        poly_features = []
        feature_names = []
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                if i <= j:
                    feature_names.append(var_names[i] + "_times_" + var_names[j])
                    poly_features.append(X[:, i] * X[:, j])

        poly_features = np.vstack(poly_features).T
        feature_names = np.array(feature_names)

        if self.n_to_select == "auto":
            self.n_to_select = int(X.shape[1] / 10)

        self.clf.fit(X, y)
        y_residuals = (self.clf.predict(X) - y) ** 2

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            coefs = np.corrcoef(
                np.hstack((poly_features, y_residuals.reshape((-1, 1)))), rowvar=False)
        correlations = coefs[-1][:-1]

        # also consider weight on y
        coefs = np.corrcoef(np.hstack((poly_features, y_residuals.reshape(-1, 1))), rowvar=False)
        correlations = 2 * correlations + coefs[-1][:-1]

        indices = np.argsort(-np.abs(correlations))

        indices = indices[np.logical_not(np.isnan(indices))][:self.n_to_select]

        self.best_coefs = correlations[indices]
        self.feature_indices = indices
        self.feature_names = feature_names[indices]
        return self
