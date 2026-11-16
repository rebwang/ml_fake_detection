"""Deeper analysis of the labeled live Instagram sample.

  python src/analyze_live.py

Writes to results/:
  live_ci.csv               stratified bootstrap 95% CIs (precision, recall, F1, PR-AUC) at 0.5
  live_pr_curves.png        PR curves on live data and on the Kaggle test set
  live_error_analysis.csv   false positives of two models vs live genuine / fake and Kaggle classes
  live_vs_kaggle_shift.csv  feature distribution shift, live vs Kaggle (per class)
  live_gone_scores.csv      model scores for accounts marked `gone` (if any)
and refreshes the auto-generated tables in RESULTS.md between the live-auto markers.
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score)

from features import FEATURES, LABEL

ERROR_MODELS = ["balanced_xgb", "imbalanced_logreg"]
NAMES = {"logreg": "LogReg", "rf": "RF", "xgb": "XGBoost"}
CFG = {"balanced": "50/50", "imbalanced": "90/10"}
START, END = "<!-- live-auto-start -->", "<!-- live-auto-end -->"


def load_models(model_dir):
    models = {}
    for path in sorted(Path(model_dir).glob("*.joblib")):
        meta = json.loads(path.with_name(path.stem + ".meta.json").read_text())
        models[path.stem] = (joblib.load(path), meta)
    return models


def point_metrics(y, prob, t=0.5):
    pred = (prob >= t).astype(int)
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "pr_auc": average_precision_score(y, prob)}


def bootstrap_ci(y, probs, n_boot, seed):
    """Stratified bootstrap: resample fakes and genuines separately, same indices for all models."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    draws = {k: [] for k in probs}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, p in probs.items():
            draws[k].append(point_metrics(y[idx], p[idx]))
    rows = []
    for k, p in probs.items():
        est = point_metrics(y, p)
        d = pd.DataFrame(draws[k])
        for m in est:
            lo, hi = np.percentile(d[m], [2.5, 97.5])
            rows.append({"model": k, "metric": m, "estimate": est[m], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def describe(df):
    """Summary statistics used for the error-analysis and shift tables."""
    return {
        "n": len(df),
        "median #posts": df["#posts"].median(),
        "median #followers": df["#followers"].median(),
        "median #follows": df["#follows"].median(),
        "median follows/followers": (df["#follows"] / df["#followers"].clip(lower=1)).median(),
        "median description length": df["description length"].median(),
        "median fullname words": df["fullname words"].median(),
        "% private": 100 * df["private"].mean(),
        "% no profile pic": 100 * (1 - df["profile pic"].mean()),
        "% external URL": 100 * df["external URL"].mean(),
        "% name==username": 100 * df["name==username"].mean(),
        "% username has digits": 100 * (df["nums/length username"] > 0).mean(),
        "mean digit share username": df["nums/length username"].mean(),
        "% 0 posts": 100 * (df["#posts"] == 0).mean(),
        "% empty bio": 100 * (df["description length"] == 0).mean(),
    }


def label(stem):
    cfg, name = stem.split("_", 1)
    return f"{NAMES[name]} ({CFG[cfg]})"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="data/live_labeled.csv")
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.results)
    out.mkdir(exist_ok=True)

    raw = pd.read_csv(a.live)
    live = raw.dropna(subset=[LABEL]).reset_index(drop=True)
    live[LABEL] = live[LABEL].astype(int)
    y = live[LABEL].to_numpy()
    kag_train = pd.read_csv(Path(a.data) / "train.csv")
    kag_test = pd.read_csv(Path(a.data) / "test.csv")
    models = load_models(a.models)
    print(f"live: {len(live)} labeled ({y.sum()} fake, {y.mean():.1%}); "
          f"{len(raw) - len(live)} unlabeled; models: {list(models)}")

    probs = {k: m.predict_proba(live[FEATURES])[:, 1] for k, (m, _) in models.items()}
    kprobs = {k: m.predict_proba(kag_test[FEATURES])[:, 1] for k, (m, _) in models.items()}

    # 1) bootstrap CIs ---------------------------------------------------------------
    ci = bootstrap_ci(y, probs, a.n_boot, a.seed)
    ci.round(3).to_csv(out / "live_ci.csv", index=False)

    # 2) PR curves ---------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"logreg": "#1f77b4", "rf": "#2ca02c", "xgb": "#d62728"}
    for ax, (title, yt, pr) in zip(axes, [
            (f"Live Instagram sample (n={len(y)}, {y.mean():.1%} fake)", y, probs),
            (f"Kaggle test set (n={len(kag_test)}, {kag_test[LABEL].mean():.0%} fake)",
             kag_test[LABEL].to_numpy(), kprobs)]):
        for k, p in pr.items():
            cfg, name = k.split("_", 1)
            prec, rec, _ = precision_recall_curve(yt, p)
            ax.step(rec, prec, where="post", color=colors[name],
                    ls="-" if cfg == "balanced" else "--",
                    label=f"{label(k)}  AP={average_precision_score(yt, p):.2f}")
        ax.axhline(yt.mean(), color="grey", ls=":", label=f"base rate {yt.mean():.1%}")
        ax.set(title=title, xlabel="Recall", xlim=(0, 1.01), ylim=(0, 1.02))
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Precision")
    fig.suptitle("Precision–recall: solid = trained 50/50, dashed = trained 90/10")
    fig.tight_layout()
    fig.savefig(out / "live_pr_curves.png", dpi=150)
    plt.close(fig)

    # 3) error analysis + shift -------------------------------------------------------
    groups = {}
    for k in ERROR_MODELS:
        pred = probs[k] >= 0.5
        groups[f"FP {k}"] = live[pred & (y == 0)]
        groups[f"TN {k}"] = live[~pred & (y == 0)]
    groups["live genuine (all)"] = live[y == 0]
    groups["live fake"] = live[y == 1]
    groups["Kaggle train genuine"] = kag_train[kag_train[LABEL] == 0]
    groups["Kaggle train fake"] = kag_train[kag_train[LABEL] == 1]
    err = pd.DataFrame({g: describe(d) for g, d in groups.items()})
    err.index.name = "statistic"
    err.round(2).to_csv(out / "live_error_analysis.csv")

    shift_groups = {
        "Kaggle genuine": kag_train[kag_train[LABEL] == 0],
        "live genuine": live[y == 0],
        "Kaggle fake": kag_train[kag_train[LABEL] == 1],
        "live fake": live[y == 1],
    }
    shift = pd.DataFrame({g: describe(d) for g, d in shift_groups.items()})
    shift.index.name = "statistic"
    shift.round(2).to_csv(out / "live_vs_kaggle_shift.csv")

    # 4) gone accounts ------------------------------------------------------------------
    gone_col = "gone" if "gone" in raw.columns else None
    gone_md = "No `gone` column in the live file."
    if gone_col is None and "status" in raw.columns:
        raw["gone"] = (raw["status"] == "gone").astype(int)
        gone_col = "gone"
    if gone_col:
        gone = raw[raw[gone_col].fillna(0).astype(float) == 1]
        if len(gone):
            gp = {k: m.predict_proba(gone[FEATURES])[:, 1] for k, (m, _) in models.items()}
            g = pd.DataFrame([{"model": label(k), "mean_p_fake": p.mean(),
                               "share_flagged_0.5": (p >= 0.5).mean()} for k, p in gp.items()])
            g.round(3).to_csv(out / "live_gone_scores.csv", index=False)
            n_lab = gone[LABEL].notna().sum()
            gone_md = (f"{len(gone)} account(s) marked `gone` ({n_lab} labeled). "
                       "Model scores:\n\n" + md_table(g.round(2)))
        else:
            gone_md = "No accounts marked `gone` yet."

    # console + RESULTS.md tables -------------------------------------------------------
    summ = pd.read_csv(out / "summary.csv")
    rows = []
    for k, (_, meta) in models.items():
        cfg, name = k.split("_", 1)
        s = summ[(summ.model == name) & (summ.trained_on == cfg) &
                 (summ.test_set == "imbalanced_test") & (summ.threshold_type == "default_0.5")].iloc[0]
        c = ci[ci.model == k].set_index("metric")
        f = lambda m: f"{c.loc[m, 'estimate']:.2f} [{c.loc[m, 'ci_low']:.2f}–{c.loc[m, 'ci_high']:.2f}]"
        fp = int(((probs[k] >= 0.5) & (y == 0)).sum())
        rows.append({"Model": label(k),
                     "Kaggle 90/10 P / R / F1": f"{s.precision_mean:.2f} / {s.recall_mean:.2f} / {s.f1_mean:.2f}",
                     "Kaggle 90/10 PR-AUC": f"{s.pr_auc_mean:.2f}",
                     "Live precision": f("precision"), "Live recall": f("recall"),
                     "Live F1": f("f1"), "Live PR-AUC": f("pr_auc"),
                     "Live FP / genuine": f"{fp} / {(y == 0).sum()}"})
    side = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(side.to_string(index=False))
    print(err.round(2).to_string())
    print(shift.round(2).to_string())
    print(gone_md)

    shift_rows = ["n", "median #posts", "median #followers", "median #follows",
                  "median follows/followers", "% private", "% no profile pic", "% external URL",
                  "% username has digits", "mean digit share username", "median description length",
                  "% 0 posts"]
    fmt = lambda v: f"{v:.0f}" if abs(v) >= 10 or v == int(v) else f"{v:.2f}"
    st = shift.loc[shift_rows].map(fmt).reset_index()
    err_rows = ["n", "median #posts", "median #followers", "median #follows", "% private",
                "% no profile pic", "% external URL", "% username has digits", "% empty bio"]
    et = err.loc[err_rows, [f"FP {k}" for k in ERROR_MODELS] +
                 ["live genuine (all)", "live fake", "Kaggle train genuine", "Kaggle train fake"]]
    et = et.map(fmt).reset_index()

    auto = "\n\n".join([
        START,
        f"**Live vs Kaggle, default threshold 0.5.** Live: {len(y)} labeled profiles, {y.sum()} fake "
        f"({y.mean():.1%}). Brackets are stratified bootstrap 95% CIs ({a.n_boot} resamples). "
        "Kaggle columns are the mean over the 10 90/10 test samples.",
        md_table(side),
        "**Feature shift, live vs Kaggle (Kaggle = train.csv).**",
        md_table(st),
        f"**False positives vs the true classes** (FP = genuine accounts flagged at 0.5).",
        md_table(et),
        "**Accounts that disappeared after collection.** " + gone_md,
        "Files: `live_ci.csv`, `live_pr_curves.png`, `live_error_analysis.csv`, "
        "`live_vs_kaggle_shift.csv`.",
        END,
    ])
    md_path = out / "RESULTS.md"
    md = md_path.read_text()
    if START in md and END in md:
        md = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: auto, md, flags=re.S)
        md_path.write_text(md)
        print("RESULTS.md: refreshed auto tables")
    else:
        (out / "live_tables.md").write_text(auto + "\n")
        print("RESULTS.md has no live-auto markers; wrote tables to live_tables.md")


if __name__ == "__main__":
    main()
