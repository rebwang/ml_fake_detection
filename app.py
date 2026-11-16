"""Streamlit risk dashboard: compares the balanced- and imbalanced-trained models.

  streamlit run app.py
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# The 11 features, named as in the Kaggle "Instagram Fake Spammer Genuine Accounts" files
# (same definitions as notebooks/01_train_models.ipynb and 02_collect_live_data.ipynb).
FEATURES = [
    "profile pic", "nums/length username", "fullname words", "nums/length fullname",
    "name==username", "description length", "external URL", "private",
    "#posts", "#followers", "#follows",
]
LABEL = "fake"
COUNT_FEATURES = ["description length", "#posts", "#followers", "#follows"]
OTHER = [f for f in FEATURES if f not in COUNT_FEATURES]  # column order after preprocessing


def _digit_ratio(s: str) -> float:
    s = s or ""
    return round(sum(c.isdigit() for c in s) / len(s), 2) if s else 0.0


def profile_to_features(username, full_name, biography, external_url,
                        is_private, posts, followers, follows, has_profile_pic) -> dict:
    """Turn raw public profile metadata into the 11 dataset features."""
    full_name = (full_name or "").strip()
    norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    return {
        "profile pic": int(bool(has_profile_pic)),
        "nums/length username": _digit_ratio(username),
        "fullname words": len(full_name.split()),
        "nums/length fullname": _digit_ratio(full_name),
        "name==username": int(bool(full_name) and norm(full_name) == norm(username)),
        "description length": len(biography or ""),
        "external URL": int(bool(external_url)),
        "private": int(bool(is_private)),
        "#posts": int(posts or 0),
        "#followers": int(followers or 0),
        "#follows": int(follows or 0),
    }


def evaluate(y_true, prob, threshold=0.5) -> dict:
    """Precision / recall / confusion counts for the fake class at a threshold."""
    y_true = np.asarray(y_true)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

MODELS, DATA, RESULTS = Path("models"), Path("data"), Path("results")
NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}

st.set_page_config(page_title="Fake Account Risk", layout="wide")


@st.cache_resource
def load_models():
    out = {}
    for p in sorted(MODELS.glob("*.joblib")):
        meta = json.loads(p.with_name(p.stem + ".meta.json").read_text())
        out[(meta["config"], meta["model"])] = (joblib.load(p), meta)
    return out


def contributions(model, name, x: pd.DataFrame) -> pd.Series:
    """Per-feature contribution to this prediction (log-odds scale where available)."""
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    z = prep.transform(x)
    cols = COUNT_FEATURES + OTHER  # column order produced by the preprocessing step
    if name == "logreg":
        vals = clf.coef_[0] * z[0]
    elif name == "xgb":
        import xgboost as xgb
        vals = clf.get_booster().predict(xgb.DMatrix(z), pred_contribs=True)[0][:-1]
    else:  # random forest: global importances only
        vals = clf.feature_importances_
    return pd.Series(vals, index=cols).sort_values(key=np.abs, ascending=False)


models = load_models()
if not models:
    st.error("No saved models found. Run `notebooks/01_train_models.ipynb` first.")
    st.stop()

st.title("Fake account risk score")
st.caption("Each account is scored by the model trained on 50/50 data and the model trained on 90/10 data.")

model_name = st.sidebar.selectbox("Model", list(NAMES), format_func=NAMES.get, index=2)
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)

tab_score, tab_perf = st.tabs(["Score an account", "Threshold trade-off"])

with tab_score:
    mode = st.radio("Input", ["Profile details", "Raw feature values"], horizontal=True)
    if mode == "Profile details":
        c1, c2 = st.columns(2)
        username = c1.text_input("Username", "john_smith1998")
        full_name = c1.text_input("Full name", "John Smith")
        bio = c1.text_area("Bio", "")
        url = c1.text_input("External URL", "")
        has_pic = c2.checkbox("Has a profile picture", True)
        private = c2.checkbox("Private account", False)
        posts = c2.number_input("Posts", 0, value=12)
        followers = c2.number_input("Followers", 0, value=150)
        follows = c2.number_input("Following", 0, value=300)
        feats = profile_to_features(username, full_name, bio, url, private,
                                    posts, followers, follows, has_pic)
    else:
        cols = st.columns(4)
        feats = {f: cols[i % 4].number_input(f, value=0.0) for i, f in enumerate(FEATURES)}
    x = pd.DataFrame([feats])[FEATURES]

    out = st.columns(2)
    for col, config in zip(out, ["balanced", "imbalanced"]):
        if (config, model_name) not in models:
            col.warning(f"No {config} model saved")
            continue
        model, meta = models[(config, model_name)]
        p = float(model.predict_proba(x)[0, 1])
        col.subheader(f"Trained on {'50/50' if config == 'balanced' else '90/10'} data")
        col.metric("Risk score (P(fake))", f"{p:.2f}")
        col.write("**Flagged as fake**" if p >= threshold else "Not flagged")
        col.caption(f"Tuned threshold for this model: {meta['tuned_threshold']:.2f}")
        contrib = contributions(model, model_name, x).head(6)
        col.bar_chart(contrib.rename("contribution"), horizontal=True)
    with st.expander("Feature values used"):
        st.dataframe(x.T.rename(columns={0: "value"}))

with tab_perf:
    test_path = DATA / "test.csv"
    if not test_path.exists():
        st.info("Put test.csv in data/ to see threshold curves.")
    else:
        test = pd.read_csv(test_path)
        seed_share = st.select_slider("Fake share in the evaluation set", [0.5, 0.25, 0.1, 0.05], 0.1)
        genuine = test[test[LABEL] == 0]
        n_fake = max(1, round(len(genuine) * seed_share / (1 - seed_share)))
        evalset = pd.concat([genuine, test[test[LABEL] == 1].sample(n=min(n_fake, (test[LABEL] == 1).sum()),
                                                                  random_state=0)])
        rows = []
        for config in ["balanced", "imbalanced"]:
            if (config, model_name) not in models:
                continue
            model, _ = models[(config, model_name)]
            prob = model.predict_proba(evalset[FEATURES])[:, 1]
            for t in np.linspace(0.05, 0.95, 19):
                m = evaluate(evalset[LABEL], prob, t)
                rows.append({"trained_on": config, "threshold": round(t, 2),
                             "precision": m["precision"], "recall": m["recall"]})
        curves = pd.DataFrame(rows)
        st.caption(f"Evaluation set: {len(genuine)} genuine and {n_fake} fake test accounts")
        c1, c2 = st.columns(2)
        c1.write("Precision by threshold")
        c1.line_chart(curves.pivot(index="threshold", columns="trained_on", values="precision"))
        c2.write("Recall by threshold")
        c2.line_chart(curves.pivot(index="threshold", columns="trained_on", values="recall"))
    for f in ["summary.csv", "live_results.csv"]:
        if (RESULTS / f).exists():
            st.write(f"**{f}**")
            st.dataframe(pd.read_csv(RESULTS / f))
