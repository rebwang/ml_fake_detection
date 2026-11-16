"""Model definitions: each is a full sklearn Pipeline so a saved file scores raw features."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from features import COUNT_FEATURES, FEATURES

OTHER = [f for f in FEATURES if f not in COUNT_FEATURES]


def _log_scale():
    # Counts are heavily skewed (followers range from 0 to millions), so log1p then scale.
    return ColumnTransformer([
        ("counts", Pipeline([("log", FunctionTransformer(np.log1p)),
                             ("scale", StandardScaler())]), COUNT_FEATURES),
        ("other", StandardScaler(), OTHER),
    ])


def _log_only():
    return ColumnTransformer([
        ("counts", FunctionTransformer(np.log1p), COUNT_FEATURES),
        ("other", "passthrough", OTHER),
    ])


def build(name: str, imbalanced: bool, pos_weight: float = 1.0, seed: int = 0):
    """Return (pipeline, hyperparameter grid). Imbalanced configs reweight the fake class."""
    cw = "balanced" if imbalanced else None
    if name == "logreg":
        pipe = Pipeline([("prep", _log_scale()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight=cw))])
        grid = {"clf__C": [0.01, 0.1, 1, 10]}
    elif name == "rf":
        pipe = Pipeline([("prep", _log_only()),
                         ("clf", RandomForestClassifier(n_estimators=300, class_weight=cw,
                                                        random_state=seed, n_jobs=-1))])
        grid = {"clf__max_depth": [None, 6, 10], "clf__min_samples_leaf": [1, 3, 5]}
    elif name == "xgb":
        pipe = Pipeline([("prep", _log_only()),
                         ("clf", XGBClassifier(n_estimators=300, eval_metric="logloss",
                                               scale_pos_weight=pos_weight if imbalanced else 1.0,
                                               random_state=seed, n_jobs=-1))])
        grid = {"clf__max_depth": [2, 3, 5], "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.8, 1.0]}
    else:
        raise ValueError(name)
    return pipe, grid


MODEL_NAMES = ["logreg", "rf", "xgb"]
