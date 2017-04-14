
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer


def f1_weighted_score(*args, **kwargs):
    return f1_score(average="weighted", *args, **kwargs)


def mse(*args, **kwargs):
    return -mean_squared_error(*args, **kwargs)


f1_weighted_scorer = make_scorer(f1_weighted_score)
mse_scorer = make_scorer(mse)
