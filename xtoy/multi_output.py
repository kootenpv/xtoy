
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier


class MOR(MultiOutputRegressor):
    def get_params(self, deep=False):
        params = super(MOR, self).get_params(deep=deep)
        params = {k.replace("estimator__", ""): v for k, v in params.items()}
        return params


class MOC(MultiOutputClassifier):
    def get_params(self, deep=False):
        params = super(MOR, self).get_params(deep=deep)
        params = {k.replace("estimator__", ""): v for k, v in params.items()}
        return params
