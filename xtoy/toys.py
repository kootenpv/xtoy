import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from xtoy.evolutionary_search import EvolutionaryAlgorithmSearchCV as evo_search
from xtoy.prep import Sparsify
from xtoy.classifiers import pick
from xtoy.classifiers import classification_or_regression


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
            ("scaler", Normalizer()),
            ('clf', self.clf())
        ])

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.X = X
        self.y = y
        if self.scoring is None:
            tp = classification_or_regression(self.y)
            self.scoring = ['f1_weighted', None][tp != 'classification']
        print(self.scoring)
        self.pick_model()
        self.get_pipeline()
        unique_combinations = np.prod(list(map(len, self.grid.values())))
        print("unique_combinations", unique_combinations)
        if 'population_size' not in self.kwargs:
            self.kwargs['population_size'] = np.clip(int(unique_combinations / 1000), 5, 10)
        if 'generations_number' not in self.kwargs:
            self.kwargs['generations_number'] = np.clip(int(unique_combinations / 200), 10, 50)

        self.evo = evo_search(self.pipeline, self.grid, scoring=self.scoring,
                              cv=self.cv, n_jobs=self.n_jobs, **self.kwargs)
        self.evo.fit(X, y)
        return self.evo.best_estimator_

    def predict(self, X):
        return self.evo.predict(X)

    def score(self, X, y):
        return self.evo.best_estimator_.score(X, y)

    def best_model_pickle(self):
        return pickle.dumps(self.evo.best_estimator_)
