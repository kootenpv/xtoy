import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from xtoy.evolutionary_search import EvolutionaryAlgorithmSearchCV as evo_search
from xtoy.prep import Sparsify
from xtoy.classifiers import pick
from xtoy.classifiers import classification_or_regression

from xtoy.scorers import f1_weighted_scorer
from xtoy.scorers import mse_scorer

from xtoy.utils import get_cv_splits

try:
    import pickle
except (ValueError, SystemError, ImportError):
    pass


class Toy:

    def __init__(self, cv=get_cv_splits, scoring=None, n_jobs=1, cl_or_reg=None, **kwargs):
        self.cv = get_cv_splits
        self.clf = None
        self.pipeline = None
        self.evo = None
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.grid = None
        self.cl_or_reg = cl_or_reg
        self.sparsify = Sparsify()

    def pick_model(self, X, y):
        chosen_model = pick(X, y, self.cl_or_reg)
        self.clf = chosen_model['clf']
        self.grid = chosen_model['grid']

    def get_pipeline(self):
        self.pipeline = Pipeline(steps=[
            #('tsvd', TruncatedSVD()),  # this one also has to have % top features chosen
            #('feature_selection', SelectFromModel(Ridge())),
            ("scaler", Normalizer()),
            ('clf', self.clf())
        ])

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = pd.DataFrame(self.sparsify.fit_transform(X).A)
        if self.scoring is None:
            tp = classification_or_regression(y)
            self.scoring = [f1_weighted_scorer, mse_scorer][tp != 'classification']
        print(self.scoring)
        self.pick_model(X, y)
        self.get_pipeline()
        unique_combinations = np.prod(list(map(len, self.grid.values())))
        print("unique_combinations", unique_combinations)
        if 'population_size' not in self.kwargs:
            self.kwargs['population_size'] = np.clip(int(unique_combinations / 1000), 5, 10)
        if 'generations_number' not in self.kwargs:
            self.kwargs['generations_number'] = np.clip(int(unique_combinations / 20), 10, 50)

        self.evo = evo_search(self.pipeline, self.grid, scoring=self.scoring,
                              cv=self.cv, n_jobs=self.n_jobs, **self.kwargs)
        self.evo.fit(X, y)
        return self.evo.best_estimator_

    def predict(self, X):
        X = self.sparsify.transform(X).A
        return self.evo.predict(X)

    def score(self, X, y):
        X = self.sparsify.transform(X).A
        return self.evo.best_estimator_.score(X, y)

    # def baselines():
    #     f1_weighted_score

    def best_model_pickle(self):
        return pickle.dumps(self.evo.best_estimator_)
