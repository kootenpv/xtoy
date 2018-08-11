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
    def __init__(
        self, cv=get_cv_splits, scoring=None, n_jobs=1, cl_or_reg=None, sparse=True, **kwargs
    ):
        self.cv = get_cv_splits
        self.evo = None
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.cl_or_reg = cl_or_reg
        self.sparsify = Sparsify(sparse=True)
        self._feature_name = None
        self.evos = []

    def get_models(self, X, y):
        return pick(X, y, self.cl_or_reg)

    def get_pipeline(self, clf):
        return Pipeline(
            steps=[
                # ('tsvd', TruncatedSVD()),  # this one also has to have % top features chosen
                # ('feature_selection', SelectFromModel(Ridge())),
                ("scaler", Normalizer()),
                ("clf", clf()),
            ]
        )

    def fit(self, X, y):
        evos = []
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._feature_name = np.array(list(X.columns))
        if isinstance(y, pd.DataFrame):
            y = np.array(y)
        if len(y.shape) > 1 and y.shape[1] == 1:
            y = y.ravel()
        X = pd.DataFrame(self.sparsify.fit_transform(X).A)
        if self.scoring is None:
            tp = classification_or_regression(y)
            self.scoring = [f1_weighted_scorer, mse_scorer][tp != "classification"]
        print(self.scoring)

        for model in self.get_models(X, y):
            clf = model["clf"]
            grid = model["grid"]
            pipeline = self.get_pipeline(clf)
            unique_combinations = np.prod(list(map(len, grid.values())))
            print("unique_combinations", unique_combinations)
            kwargs = self.kwargs.copy()
            if "population_size" not in self.kwargs:
                kwargs["population_size"] = np.clip(int(unique_combinations / 1000), 5, 10)
            if "generations_number" not in kwargs:
                kwargs["generations_number"] = np.clip(int(unique_combinations / 20), 10, 50)

            evo = evo_search(
                pipeline, grid, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs, **kwargs
            )
            evo.fit(X, y)
            evos.append((evo.best_score_, evo))

        self.evos = evos
        self.best_evo = sorted(self.evos, key=lambda x: x[0])[-1][1]
        return self.best_evo.best_estimator_

    def predict(self, X):
        X = self.sparsify.transform(X).A
        return self.best_evo.predict(X)

    def predict_proba(self, X):
        X = self.sparsify.transform(X).A
        return self.best_evo.predict_proba(X)

    def score(self, X, y):
        X = self.sparsify.transform(X).A
        return self.best_evo.best_estimator_.score(X, y)

    # def baselines():
    #     f1_weighted_score

    def best_model_pickle(self):
        return pickle.dumps(self.best_evo.best_estimator_)

    @property
    def feature_importances_(self):
        clf = self.best_evo.best_estimator_.steps[-1][1]
        if hasattr(clf, "feature_importances_"):
            weights = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            weights = np.abs(clf.coef_)
        else:
            raise ValueError("No importances could be computed (requires a different classifier).")
        return weights

    @property
    def feature_names_(self):
        return self.sparsify.feature_names_

    @property
    def feature_indices_(self):
        return self.sparsify.feature_indices_

    def best_features_(self, n=10, aggregation=np.max):
        # a bit annoying that aggregation makes different shape if aggregation=None
        # this is whether interested in original features, or post processing.
        # maybe split this in 2 functions
        if aggregation is None:
            data = list(zip(self.feature_importances_, self.feature_names_))
        else:
            pdata = pd.DataFrame(
                {
                    "features": self._feature_name[self.feature_indices_],
                    "importances": self.feature_importances_,
                }
            )
            agg = pdata.groupby(["features"]).agg(aggregation)
            data = list(zip(agg["importances"].values, agg.index))
        return sorted(data)[-n:]
