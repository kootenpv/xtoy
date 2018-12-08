[![Build Status](https://travis-ci.org/kootenpv/xtoy.svg?branch=master)](https://travis-ci.org/kootenpv/xtoy)

## XtoY

`pip install xtoy`

Go from 'X' to 'y' without effort.

``` python
from sklearn.datasets import load_diabetes
from xtoy.toys import Toy
X, y = load_diabetes(return_X_y=True)
toy = Toy()
toy.fit(X[:300], y[:300])
toy.predict(X[300:])
```

#### Tries to minimize time-to-first-model

And a reasonable one at that.

Check how important each variable is:

```python
# names of variables are numbers - only in this example - otherwise usually strings
toy.best_features_()
[(0.02541263748358529, 4),
 (0.03964045497300279, 6),
 (0.04000655539791701, 5),
 (0.047171804294566556, 0),
 (0.05355633793403717, 1),
 (0.05598481754558562, 9),
 (0.06349342396487742, 3),
 (0.09050228976499292, 7),
 (0.28327316154993126, 2),
 (0.3009585170915041, 8)]
```

For further inspection, have a look at the pipeline and how important each variable is:

```python
# toy.best_pipeline_
```

#### Guarantee

The goal will be to accept ANY data and come up with a "sensible" prediction.

If your dataset *doesn't* work (asymptotically not happening), [post an issue](https://github.com/kootenpv/xtoy/issues).

#### Test driven

Quality guarantee by testing code changes, with loss measurements on lots of data problems.

#### Features

- ✓ Takes care of encoding text, categorical, dates (several features), continuous
- Considers data size (small data -> feature engineering, big data -> feature selection)
- ✓ Takes care of missing values
- ✓ Creates a model
- ✓ Optimizes model parameters
- ✓ Gives you a first prediction
- ✓ Contains a `RegexVectorizer`

#### Roadmap

- More customizability
- Tree-based data (being able to exclude grouped variables quickly)
