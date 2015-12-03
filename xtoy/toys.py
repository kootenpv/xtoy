from sklearn.pipeline import Pipeline

from xtoy.evolutionary_search import EvolutionaryAlgorithmSearchCV as evo_search
from xtoy.prep import Sparsify
from xtoy.classifiers import pick
from xtoy.classifiers import classification_or_regression
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier

import numpy as np

try:
    import pickle
except (ValueError, SystemError, ImportError):
    pass


class Toy:

    def __init__(self, cv=10, scoring=None, n_jobs=1, cl_or_reg=None, **kwargs):
        self.cv = cv
        self.clf = None
        self.pipeline = None
        self.evo = None
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.X = None
        self.y = None
        self.grid = None
        self.cl_or_reg = cl_or_reg

    def pick_model(self):
        chosen_model = pick(self.X, self.y, self.cl_or_reg)
        self.clf = chosen_model['clf']
        self.grid = chosen_model['grid']

    def get_pipeline(self):
        self.pipeline = Pipeline(steps=[
            ('sparse', Sparsify()),
            #('tsvd', TruncatedSVD()),  # this one also has to have % top features chosen
            #('feature_selection', SelectFromModel(Ridge())),
            ('clf', self.clf())
        ])

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.scoring is None:
            tp = classification_or_regression(self.y)
            # MULTIOUTPUT NOT SUPPORTED FOR REGRESSION (probably DEAPs fault)
            self.scoring = ['f1_weighted', 'r2'][tp != 'classification']
        self.pick_model()
        self.get_pipeline()
        unique_combinations = np.prod(list(map(len, self.grid.values())))
        if 'population_size' not in self.kwargs:
            self.kwargs['population_size'] = np.clip(int(unique_combinations / 1000), 3, 10)
        if 'generations_number' not in self.kwargs:
            self.kwargs['generations_number'] = np.clip(int(unique_combinations / 200), 0, 50)

        self.evo = evo_search(self.pipeline, self.grid, cv=self.cv,
                              scoring=self.scoring, n_jobs=self.n_jobs, **self.kwargs)
        m = 0
        max_tries = 10
        while not self.evo.best_params_ and m < max_tries:
            print('fitting', m)
            f = self.evo.fit(X, y)
            m += 1
        return f

    def predict(self, X):
        return self.evo.predict(X)

    def score(self, X, y):
        return self.evo.best_estimator_.score(X, y)

    def best_model_pickle(self):
        return pickle.dumps(self.evo.best_estimator_)
