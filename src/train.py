"""Train and save models for two configurations and cross-evaluate them.

  balanced    : Kaggle train.csv / test.csv as given (50/50 genuine vs fake).
  imbalanced  : same genuine accounts, fake accounts downsampled to ~10% in both
                train and test. Repeated over several seeds because only a few
                dozen fake accounts remain; the seed-0 models are the ones saved.

Usage:  python src/train.py --data data --out models --results results
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from features import FEATURES, LABEL
from metrics import best_f1_threshold, evaluate
from models import MODEL_NAMES, build

warnings.filterwarnings("ignore", category=UserWarning)


def load(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    missing = [c for c in FEATURES + [LABEL] if c not in train.columns]
    if missing:
        raise SystemExit(f"train.csv is missing columns: {missing}")
    n_dup = int(train.duplicated().sum())
    if n_dup:
        print(f"Dropping {n_dup} exact duplicate rows from train.csv")
        train = train.drop_duplicates().reset_index(drop=True)
    return train, test


def make_imbalanced(df: pd.DataFrame, fake_share: float, seed: int) -> pd.DataFrame:
    """Keep every genuine account and sample fakes so they make up `fake_share`."""
    genuine = df[df[LABEL] == 0]
    n_fake = max(1, round(len(genuine) * fake_share / (1 - fake_share)))
    fake = df[df[LABEL] == 1].sample(n=n_fake, random_state=seed)
    return pd.concat([genuine, fake]).sample(frac=1, random_state=seed).reset_index(drop=True)


def fit(name, train, imbalanced, seed):
    X, y = train[FEATURES], train[LABEL]
    pos_weight = (y == 0).sum() / max(1, (y == 1).sum())
    pipe, grid = build(name, imbalanced, pos_weight, seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=-1)
    search.fit(X, y)
    # Out-of-fold probabilities pick a tuned threshold without touching the test set.
    oof = cross_val_predict(search.best_estimator_, X, y, cv=cv, method="predict_proba")[:, 1]
    return search.best_estimator_, search.best_params_, best_f1_threshold(y, oof)


def save(model, name, config, params, threshold, train, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{config}_{name}"
    joblib.dump(model, out / f"{stem}.joblib")
    if name == "xgb":  # native XGBoost weights as well
        model.named_steps["clf"].save_model(out / f"{stem}.json")
    meta = {"model": name, "config": config, "features": FEATURES,
            "n_train": len(train), "fake_share_train": float(train[LABEL].mean()),
            "best_params": params, "tuned_threshold": threshold}
    (out / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--fake-share", type=float, default=0.10)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    out, res = Path(a.out), Path(a.results)
    res.mkdir(parents=True, exist_ok=True)

    train, test = load(Path(a.data))
    print(f"train {len(train)} rows ({train[LABEL].mean():.0%} fake), "
          f"test {len(test)} rows ({test[LABEL].mean():.0%} fake)")

    rows = []

    def score(model, name, config, seed, thr, test_name, tdf):
        prob = model.predict_proba(tdf[FEATURES])[:, 1]
        for kind, t in [("default_0.5", 0.5), ("tuned", thr)]:
            rows.append({"model": name, "trained_on": config, "seed": seed,
                         "test_set": test_name, "threshold_type": kind,
                         **evaluate(tdf[LABEL], prob, t)})

    imb_tests = {s: make_imbalanced(test, a.fake_share, s) for s in range(a.seeds)}

    # 1) balanced models: one fit, evaluated on the balanced test and every imbalanced test
    for name in MODEL_NAMES:
        model, params, thr = fit(name, train, False, 0)
        save(model, name, "balanced", params, thr, train, out)
        score(model, name, "balanced", 0, thr, "balanced_test", test)
        for s, t in imb_tests.items():
            score(model, name, "balanced", s, thr, "imbalanced_test", t)
        print(f"balanced   {name:7s} params={params} thr={thr:.2f}")

    # 2) imbalanced models: one fit per seed; seed 0 is saved
    for s in range(a.seeds):
        imb_train = make_imbalanced(train, a.fake_share, s)
        for name in MODEL_NAMES:
            model, params, thr = fit(name, imb_train, True, s)
            if s == 0:
                save(model, name, "imbalanced", params, thr, imb_train, out)
                print(f"imbalanced {name:7s} params={params} thr={thr:.2f} "
                      f"(train {len(imb_train)} rows, {imb_train[LABEL].sum()} fake)")
            score(model, name, "imbalanced", s, thr, "imbalanced_test", imb_tests[s])
            score(model, name, "imbalanced", s, thr, "balanced_test", test)

    long = pd.DataFrame(rows)
    long.to_csv(res / "metrics_all_seeds.csv", index=False)
    keys = ["model", "trained_on", "test_set", "threshold_type"]
    cols = ["precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"]
    summary = long.groupby(keys)[cols].agg(["mean", "std"]).round(3)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(res / "summary.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\nMean over seeds (fake class):")
    print(long.groupby(keys)[["precision", "recall", "f1", "pr_auc"]].mean().round(3))


if __name__ == "__main__":
    main()
