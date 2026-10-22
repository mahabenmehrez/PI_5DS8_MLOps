"""The final, external test set of team_eval: removed APIs from the libraries' own release notes.

Each candidate is one use of an API that the library's release notes list as removed; the agent must pin an
older version of that library (or report, when the team's requirements pin it). The pool was filtered by one
objective test, before sampling: the code must fail on the newest versions that install on Python 3.10 (numpy
2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3). np.bool failed that test (it exists again in numpy 2).
The faults already used in development (np.float, np.product, DataFrame.append, mean_squared_error squared,
scipy.integrate.simps) are not in the pool.

Release notes: numpy 2.0 https://numpy.org/doc/stable/release/2.0.0-notes.html, numpy 1.24
https://numpy.org/doc/stable/release/1.24.0-notes.html, pandas 2.0 https://pandas.pydata.org/docs/whatsnew/v2.0.0.html,
scikit-learn https://scikit-learn.org/stable/whats_new.html (1.2, 1.4, 1.6), scipy 1.12 / 1.14
https://docs.scipy.org/doc/scipy/release.html
"""
import random

FINAL_SEED = 20261021        # fixed before the draw; the draw is final_faults()
FINAL_SIZE = 6

POOL = {   # id: (library, code)
    "numpy_alltrue": ("numpy", "import numpy as _np\n_np.alltrue([True])\n"),
    "numpy_sometrue": ("numpy", "import numpy as _np\n_np.sometrue([True])\n"),
    "numpy_cumproduct": ("numpy", "import numpy as _np\n_np.cumproduct([1, 2])\n"),
    "numpy_asfarray": ("numpy", "import numpy as _np\n_np.asfarray([1, 2])\n"),
    "numpy_find_common_type": ("numpy", "import numpy as _np\n_np.find_common_type([_np.float32], [])\n"),
    "numpy_round_": ("numpy", "import numpy as _np\n_np.round_(1.25, 1)\n"),
    "numpy_NaN": ("numpy", "import numpy as _np\n_ = _np.NaN\n"),
    "numpy_float_": ("numpy", "import numpy as _np\n_ = _np.float_(1.0)\n"),
    "numpy_unicode_": ("numpy", "import numpy as _np\n_ = _np.unicode_('a')\n"),
    "numpy_int_alias": ("numpy", "import numpy as _np\n_ = _np.int(1)\n"),
    "numpy_object_alias": ("numpy", "import numpy as _np\n_ = _np.object\n"),
    "pandas_df_iteritems": ("pandas", "import pandas as _pd\nlist(_pd.DataFrame({'a': [1]}).iteritems())\n"),
    "pandas_series_iteritems": ("pandas", "import pandas as _pd\nlist(_pd.Series([1]).iteritems())\n"),
    "pandas_series_append": ("pandas", "import pandas as _pd\n_pd.Series([1]).append(_pd.Series([2]))\n"),
    "pandas_lookup": ("pandas", "import pandas as _pd\n_pd.DataFrame({'a': [1]}).lookup([0], ['a'])\n"),
    "pandas_is_monotonic": ("pandas", "import pandas as _pd\n_ = _pd.Index([1, 2]).is_monotonic\n"),
    "pandas_mad": ("pandas", "import pandas as _pd\n_pd.DataFrame({'a': [1.0, 2.0]}).mad()\n"),
    "pandas_read_csv_squeeze": ("pandas", "import io, pandas as _pd\n_pd.read_csv(io.StringIO('a\\n1\\n'), squeeze=True)\n"),
    "pandas_error_bad_lines": ("pandas", "import io, pandas as _pd\n"
                                         "_pd.read_csv(io.StringIO('a\\n1\\n'), error_bad_lines=False)\n"),
    "pandas_line_terminator": ("pandas", "import pandas as _pd\n_pd.DataFrame({'a': [1]}).to_csv(line_terminator='\\n')\n"),
    "pandas_datetime": ("pandas", "import pandas as _pd\n_ = _pd.datetime\n"),
    "sklearn_load_boston": ("scikit-learn", "from sklearn.datasets import load_boston as _lb\n"),
    "sklearn_plot_confusion_matrix": ("scikit-learn", "from sklearn.metrics import plot_confusion_matrix as _pcm\n"),
    "sklearn_get_feature_names": ("scikit-learn", "from sklearn.preprocessing import OneHotEncoder as _OHE\n"
                                                  "_OHE().fit([['a']]).get_feature_names()\n"),
    "sklearn_ohe_sparse": ("scikit-learn", "from sklearn.preprocessing import OneHotEncoder as _OHE\n"
                                           "_OHE(sparse=False).fit([['a']])\n"),
    "sklearn_bagging_base_estimator": ("scikit-learn", "from sklearn.ensemble import BaggingClassifier as _B\n"
                                                       "from sklearn.tree import DecisionTreeClassifier as _T\n"
                                                       "_B(base_estimator=_T()).fit([[0], [1]], [0, 1])\n"),
    "sklearn_adaboost_samme_r": ("scikit-learn", "from sklearn.ensemble import AdaBoostClassifier as _A\n"
                                                 "_A(algorithm='SAMME.R').fit([[0], [1]], [0, 1])\n"),
    "scipy_trapz": ("scipy", "from scipy.integrate import trapz as _t\n_t([1.0, 2.0])\n"),
    "scipy_cumtrapz": ("scipy", "from scipy.integrate import cumtrapz as _c\n_c([1.0, 2.0])\n"),
    "scipy_interp2d": ("scipy", "from scipy.interpolate import interp2d as _i\n_i([0, 1, 0, 1], [0, 0, 1, 1], [1, 2, 3, 4])\n"),
    "scipy_binom_test": ("scipy", "from scipy.stats import binom_test as _b\n_b(3, 10)\n"),
    "scipy_misc_face": ("scipy", "from scipy.misc import face as _f\n"),
}


def final_faults() -> list[str]:
    """The 6 faults of the final set: a seeded draw from the pool (in the pool's order, as drawn)."""
    return random.Random(FINAL_SEED).sample(sorted(POOL), FINAL_SIZE)


def code_of(fault: str) -> str:
    return POOL[fault][1]


CASCADE_SEED = 20261022      # the second draw, for the cascade test (fixed before the draw, 2026-10-21)


def cascade_faults() -> list[str]:
    """6 faults drawn from the pool minus the final set's: unseen by every earlier run."""
    rest = sorted(set(POOL) - set(final_faults()))
    return random.Random(CASCADE_SEED).sample(rest, FINAL_SIZE)
