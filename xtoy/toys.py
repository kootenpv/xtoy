import pandas as pd
import numpy as np
import scipy.stats

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from xtoy.evolutionary_search import EvolutionaryAlgorithmSearchCV as evo_search
from xtoy.prep import Featurizer
from xtoy.classifiers import pick
from xtoy.classifiers import classification_or_regression

from xtoy.scorers import f1_weighted_scorer
from xtoy.scorers import mse_scorer

from xtoy.multi_output import MOR, MOC
from xtoy.utils import get_cv_splits
from xtoy.utils import calculate_complexity

from sklearn.neighbors.base import NeighborsBase

try:
    import pickle
except (ValueError, SystemError, ImportError):
    pass


class Toy:
    """ Toy object """

    def __init__(
        self,
        cv=get_cv_splits,
        scoring=None,
        n_jobs=1,
        cl_or_reg=None,
        sparse=True,
        use_lightgbm=False,
        use_xgboost=False,
        **kwargs
    ):
        self.cv = get_cv_splits
        self.evo = None
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.cl_or_reg = cl_or_reg
        self.featurizer = Featurizer(sparse=sparse)
        self._feature_name = None
        self.evos = []
        self.use_lightgbm = use_lightgbm
        self.use_xgboost = use_xgboost

    def get_models(self, X, y):
        models = pick(X, y, self.cl_or_reg)
        models = [x for x in models if self.use_xgboost or x["name"] != "xgb"]
        models = [x for x in models if self.use_lightgbm or x["name"] != "lgb"]
        return models

    def get_pipeline(self, clf):
        return Pipeline(
            steps=[
                # ('tsvd', TruncatedSVD()),  # this one also has to have % top features chosen
                # ('feature_selection', SelectFromModel(Ridge())),
                ("scaler", Normalizer()),
                ("estimator", clf()),
            ]
        )

    def handle_multi_output(self, y, name, clf):
        y = np.array(y)
        if len(y.shape) > 1 and y.shape[1] > 1:
            if name == "xgb":
                return None
            MO = MOC if "Classif" in clf.__name__ else MOR
            return lambda: MO(clf())
        return clf

    def fit(self, X, y):
        evos = []
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._feature_name = np.array(list(X.columns))
        if isinstance(y, pd.DataFrame):
            y = np.array(y)
        elif hasattr(y, "__array__"):
            y = y.__array__()
        elif len(y.shape) > 1 and y.shape[1] == 1:
            y = y.ravel()
        X = pd.DataFrame(self.featurizer.fit_transform(X).A)
        self.cl_or_reg = self.cl_or_reg or classification_or_regression(y)
        cl_or_reg = self.cl_or_reg
        if self.scoring is None:
            self.scoring = [f1_weighted_scorer, mse_scorer][cl_or_reg != "classification"]
        print(self.scoring)

        complexity = calculate_complexity(X, y, cl_or_reg)
        for model in self.get_models(X, y):
            try:
                print("estimator:", model["name"])
                grid = model["grid"]
                if complexity > model["max_complexity"]:
                    continue
                clf = self.handle_multi_output(y, model["name"], model["clf"])
                if clf is None:
                    continue
                pipeline = self.get_pipeline(clf)
                unique_combinations = np.prod(list(map(len, grid.values())))
                print("unique_combinations", unique_combinations)
                kwargs = self.kwargs.copy()
                if "population_size" not in self.kwargs:
                    kwargs["population_size"] = np.clip(int(unique_combinations / 100), 5, 10)
                if "generations_number" not in kwargs:
                    kwargs["generations_number"] = np.clip(int(unique_combinations / 10), 10, 50)

                evo = evo_search(
                    pipeline, grid, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs, **kwargs
                )
                evo.fit(X, y)
                evos.append((evo.best_score_, evo))
            except KeyboardInterrupt:
                if not evos:
                    print("Stopped by user. No models finished trained; failed to fit.")
                    raise
                print("Stopped by user. {} models trained.".format(len(evos)))
        self.evos = evos
        self.best_evo = sorted(self.evos, key=lambda x: x[0])[-1][1]
        # import warnings
        # warnings.warn("best: {}".format(self.best_evo.best_estimator_))
        print("best: {}".format(self.best_evo.best_estimator_))
        return self.best_evo.best_estimator_

    def predict(self, X):
        X = self.featurizer.transform(X).A
        return self.best_evo.predict(X)

    def predict_proba(self, X):
        X = self.featurizer.transform(X).A
        return self.best_evo.predict_proba(X)

    def ensemble_predict(self, X):
        fn = scipy.stats.mode if self.cl_or_reg == "classification" else np.mean
        return fn([x[1].predict(X) for x in self.evos], axis=0)[0][0]

    def ensemble_importances_(self, X, y):
        for score, clf in self.evos:
            clf.estimator.fit(X, y)
        Z = [x[0] * self.feature_importances_(x[1].estimator.steps[-1][1]) for x in self.evos]
        weight_sum = sum([x[0] for x in self.evos])
        return np.sum(Z, axis=0) / weight_sum

    def score(self, X, y):
        X = self.featurizer.transform(X).A
        return self.best_evo.best_estimator_.score(X, y)

    # def baselines():
    #     f1_weighted_score

    def best_model_pickle(self):
        return pickle.dumps(self.best_pipeline_)

    def feature_importances_(self, clf):
        if hasattr(clf, "estimator"):
            clf = clf.estimator
        if hasattr(clf, "feature_importances_"):
            weights = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            weights = np.abs(clf.coef_)
        elif isinstance(clf, NeighborsBase):
            weights = np.ones(len(self.featurizer.feature_names_))
        else:
            raise ValueError("No importances could be computed (requires a different classifier).")
        weights = np.abs(weights)
        if len(weights.shape) > 1:
            weights = weights.mean(axis=0)
        weights = weights / np.sum(weights)
        assert np.isclose(weights.sum(), 1)
        return weights

    @property
    def feature_names_(self):
        return self.featurizer.feature_names_

    @property
    def feature_indices_(self):
        return self.featurizer.feature_indices_

    def best_features_(self, importances=None, n=10, aggregation=np.max):
        # a bit annoying that aggregation makes different shape if aggregation=None
        # this is whether interested in original features, or post processing.
        # maybe split this in 2 functions
        """
        Default is to use the feature importances from the best model.
        If importances is not None, it is expected to be an array with weights for the features.

        By default it will aggregate the importances for e.g. text and categorical features.
        If aggregation is set to None, it will instead print the raw importances of converted X

        Feature weights sum to 1. """
        if importances is None:
            importances = self.feature_importances_(self.best_evo.best_estimator_.steps[-1][1])
        if aggregation is None:
            data = list(zip(importances, self.feature_names_))
        else:
            pdata = pd.DataFrame(
                {"features": self._feature_name[self.feature_indices_], "importances": importances}
            )
            agg = pdata.groupby(["features"]).agg(aggregation)
            data = list(zip(agg["importances"].values, agg.index))
        return sorted(data)[-n:]

    @property
    def best_pipeline_(self):
        return self.best_evo.best_estimator_
