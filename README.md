[![Build Status](https://travis-ci.org/kootenpv/xtoy.svg?branch=master)](https://travis-ci.org/kootenpv/xtoy)

## XtoY

Go from 'X' to 'y' without effort.

``` python
from sklearn.datasets import load_digits
from xtoy.toys import Toy
X, y = load_digits(return_X_y=True)
toy = Toy()
toy.fit(X[:900], y[:900])
toy.predict(X[900:])
```

#### Guarantee

The goal will be to accept ANY data and come up with a "sensible" prediction.

If your dataset *doesn't* work (asymptotically not happening), [post an issue](https://github.com/kootenpv/xtoy/issues).

#### Test driven

Quality guarantee by testing code changes, with loss measurements on lots of data problems.

#### Features

- Takes care of encoding text, categorical, dates (not yet), continuous
- Considers data size (small data -> feature engineering, big data -> feature selection)
- Takes care of missing values
- Creates a model
- Optimizes model parameters
- Gives you a first prediction

#### Roadmap

- Adding dates
- Considering everything as sparse data, or
- Working with huge data by considering columns at a time
- Better classifier settings
- More customizability
- Tree-based data (being able to exclude grouped variables quickly)
