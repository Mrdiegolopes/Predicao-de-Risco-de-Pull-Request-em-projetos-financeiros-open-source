import sys
import pickle
import logging
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

from xgboost import XGBClassifier


# =========================
# CONFIG
# =========================
DEFAULT_X_PATH = "src/transformed/modeling/X_features.csv"
DEFAULT_Y_PATH = "src/transformed/modeling/y_target.csv"

CV_METRIC = "roc_auc"
N_SPLITS = 10
RANDOM_STATE = 0

BALANCE_MODES = ["NONE", "SMOTE"]


# =========================
# MODELS
# =========================
BASE_MODELS = {
    "LRC": LogisticRegression(max_iter=10**6, solver="liblinear"),
    "SVC": LinearSVC(max_iter=10**6),
    "RFC": RandomForestClassifier(random_state=RANDOM_STATE),
    "GBC": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "BRF": BalancedRandomForestClassifier(random_state=RANDOM_STATE),
    "HGB": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGB": XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    )
}

LINEAR_MODELS = {"LRC", "SVC"}


# =========================
# GRIDS
# =========================
PARAM_GRIDS = {
    "LRC": {
        "model__C": [0.01, 0.1, 1, 5, 10]
    },
    "SVC": {
        "model__C": [0.01, 0.1, 1, 5]
    },
    "RFC": {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 6, 10]
    },
    "GBC": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1]
    },
    "BRF": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 6, 10]
    },
    "HGB": {
        "model__max_iter": [200, 400],
        "model__learning_rate": [0.05, 0.1]
    },
    "XGB": {
        "model__n_estimators": [300, 600],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5, 7]
    }
}


# =========================
# LOG
# =========================
def setup_logger():
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    fh = logging.FileHandler("pipeline.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    logger.addHandler(fh)

    return logger


# =========================
# DATA
# =========================
def load_xy(x_path, y_path):
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).iloc[:, 0]

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = ~X.isna().any(axis=1)

    return X.loc[mask], y.loc[mask].astype(int)


# =========================
# METRICS
# =========================
def continuous_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def compute_metrics(y_true, y_pred, y_score):
    return {
        "roc_auc": metrics.roc_auc_score(y_true, y_score),
        "f1": metrics.f1_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred)
    }


# =========================
# PIPELINE FACTORY
# =========================
def make_pipeline(model_name, balance_mode):
    steps = []

    if model_name in LINEAR_MODELS:
        steps.append(("scaler", StandardScaler()))

    if balance_mode == "SMOTE":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))

    steps.append(("model", BASE_MODELS[model_name]))

    return Pipeline(steps)


# =========================
# EXPERIMENT
# =========================
def run_experiments(X, y, logger):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for balance in BALANCE_MODES:
        for model_name in BASE_MODELS.keys():

            logger.debug(f"Running {model_name} | Balance={balance}")

            pipe = make_pipeline(model_name, balance)
            grid = GridSearchCV(
                pipe,
                PARAM_GRIDS[model_name],
                scoring=CV_METRIC,
                cv=5,
                n_jobs=-1
            )

            scores = []

            for train_idx, test_idx in skf.split(X, y):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

                grid.fit(X_tr, y_tr)

                best = grid.best_estimator_
                y_pred = best.predict(X_te)
                y_score = continuous_score(best, X_te)

                scores.append(compute_metrics(y_te, y_pred, y_score))

            mean_scores = pd.DataFrame(scores).mean().to_dict()

            mean_scores.update({
                "model": model_name,
                "balance": balance
            })

            results.append(mean_scores)

            logger.debug(f"→ ROC-AUC={mean_scores['roc_auc']:.4f}")

    return pd.DataFrame(results)


# =========================
# FINAL TRAIN
# =========================
def train_champion(X, y, champion_row, logger):
    model = champion_row["model"]
    balance = champion_row["balance"]

    pipe = make_pipeline(model, balance)
    grid = GridSearchCV(
        pipe,
        PARAM_GRIDS[model],
        scoring=CV_METRIC,
        cv=5,
        n_jobs=-1
    )

    grid.fit(X, y)

    with open(f"model_{model}_{balance}.pkl", "wb") as f:
        pickle.dump(grid.best_estimator_, f)

    logger.debug(f"Saved champion: model_{model}_{balance}.pkl")
    logger.debug(f"Best params: {grid.best_params_}")


# =========================
# START
# =========================
def start(x_path, y_path):
    logger = setup_logger()

    X, y = load_xy(x_path, y_path)

    results = run_experiments(X, y, logger)
    results.to_csv("metrics_comparison.csv", index=False)

    champion = results.sort_values("roc_auc", ascending=False).iloc[0]
    logger.debug(f"Champion: {champion.to_dict()}")

    train_champion(X, y, champion, logger)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        start(sys.argv[1], sys.argv[2])
    else:
        start(DEFAULT_X_PATH, DEFAULT_Y_PATH)
