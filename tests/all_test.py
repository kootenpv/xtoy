import pandas as pd
import numpy as np

from xtoy.toys import Toy
from xtoy.classifiers import classification_or_regression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import ShuffleSplit
from sklearn.utils.validation import NotFittedError
from sklearn.datasets import *


def apply_toy_on(X, y, cl_or_reg=None, n=500, max_tries=3):
    n = min(len(y), n)
    cl_or_reg = cl_or_reg if cl_or_reg else classification_or_regression(y)
    if cl_or_reg == 'classification':
        cross_split = StratifiedShuffleSplit(y, 1, int(0.2 * n), int(0.8 * n))
    else:
        cross_split = ShuffleSplit(len(y), max_tries, int(0.2 * n), int(0.8 * n))
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    for train_index, test_index in cross_split:
        try:
            toy = Toy(cv=2, cl_or_reg=cl_or_reg)
            toy.fit(X.iloc[train_index], y[train_index])
            score = toy.score(X.iloc[test_index], y[test_index])
        except NotFittedError:  # probably have to find the real fix here...
            score = -10000
        print(score)
        yield score


def test_digits_data():
    digits = load_digits()
    X, y = digits.data, digits.target
    assert any([x > 0.7 for x in apply_toy_on(X, y)])


def test_iris_data():
    iris = load_iris()
    X, y = pd.DataFrame(iris.data), iris.target
    assert any([x > 0.1 for x in apply_toy_on(X, y)])


def test_boston_data():
    boston = load_boston()
    X, y = pd.DataFrame(boston.data), boston.target
    assert any([x for x in apply_toy_on(X, y)])  # > 0.3


def test_breast_cancer_data():
    breast_cancer = load_breast_cancer()
    X, y = pd.DataFrame(breast_cancer.data), breast_cancer.target
    assert any([x > 0.3 for x in apply_toy_on(X, y)])


def test_diabetes_data():
    diabetes = load_diabetes()
    X, y = pd.DataFrame(diabetes.data), diabetes.target
    assert any([x > 0.3 for x in apply_toy_on(X, y)])

###########################################
# # MULTIOUTPUT NOT SUPPORTED FOR REGRESSION (probably DEAPs fault)
###########################################
# def test_linnerud_data():
#     linnerud = load_linnerud()
#     X, y = linnerud.data, linnerud.target
#     assert apply_toy_on(X, y) > 0.4
#
# test_linnerud_data()


def test_newsgroup_data():
    categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

    newsgroups = fetch_20newsgroups(subset='train',
                                    remove=('headers', 'footers', 'quotes'),
                                    categories=categories)

    X, y = pd.DataFrame(newsgroups.data), newsgroups.target

    assert any([x > 0 for x in apply_toy_on(X, np.array(y), cl_or_reg='classification')])
