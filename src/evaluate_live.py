"""Score the labeled live sample with every saved model and compare side by side.

  python src/evaluate_live.py --live data/live_labeled.csv

The CSV needs the 11 feature columns plus `fake` (final label). If `labeler_a`
and `labeler_b` are filled in, inter-annotator agreement (Cohen's kappa) is reported.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from features import FEATURES, LABEL
from metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/live_labeled.csv")
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    df = pd.read_csv(a.live)

    if {"labeler_a", "labeler_b"} <= set(df.columns):
        both = df.dropna(subset=["labeler_a", "labeler_b"])
        if len(both):
            k = cohen_kappa_score(both["labeler_a"].astype(int), both["labeler_b"].astype(int))
            print(f"Cohen's kappa between labelers: {k:.3f} on {len(both)} profiles")

    df = df.dropna(subset=[LABEL])
    df[LABEL] = df[LABEL].astype(int)
    print(f"live sample: {len(df)} labeled profiles, {df[LABEL].mean():.1%} fake")

    rows = []
    for path in sorted(Path(a.models).glob("*.joblib")):
        model = joblib.load(path)
        meta = json.loads(path.with_name(path.stem + ".meta.json").read_text())
        prob = model.predict_proba(df[FEATURES])[:, 1]
        for kind, t in [("default_0.5", 0.5), ("tuned", meta.get("tuned_threshold", 0.5))]:
            rows.append({"model": meta.get("model", path.stem), "trained_on": meta.get("config"),
                         "threshold_type": kind, **evaluate(df[LABEL], prob, t)})
    res = pd.DataFrame(rows).round(3)
    Path(a.results).mkdir(exist_ok=True)
    res.to_csv(Path(a.results) / "live_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(res[["model", "trained_on", "threshold_type", "precision", "recall", "f1",
               "pr_auc", "fp", "fn"]].to_string(index=False))


if __name__ == "__main__":
    main()
