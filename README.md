# ml_fake_detection

Fake account detection: CS6140 project code.

This code covers the professor's feedback. It trains models on **50/50** data and on **90/10** data, saves the weights for each, and tests both sets of models side by side on a **live, uncurated** sample of Instagram profiles.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Put the Kaggle files `train.csv` and `test.csv` (Instagram Fake Spammer Genuine Accounts) in `data/`.

## Pipeline (run the notebooks in order)
All code lives in the three notebooks in `notebooks/`. Each is self-contained (no shared modules) and explains its steps for the report. Run them from the project root or from `notebooks/`. To re-run all three and save the outputs in place:
```bash
cd notebooks
for n in 01_train_models 02_collect_live_data 03_evaluate_live; do jupyter nbconvert --execute --to notebook --inplace "$n.ipynb"; done
```

### 1. Train and save models: `notebooks/01_train_models.ipynb`
Takes about 2 to 3 minutes.
- **Balanced (50/50):** trains on Kaggle's train.csv and tests on test.csv. Exact duplicate rows are dropped from train.csv first.
- **Imbalanced (90/10):** keeps every genuine account and randomly downsamples fake accounts to 10%, in both train and test. Train and test are built from Kaggle's own split, so they never overlap. There are only about 32 fake accounts in training and 7 in test, so this is repeated over 10 random seeds and the mean ± std is reported. The seed-0 models are the ones saved.
- **Models:** Logistic Regression, Random Forest and XGBoost. Hyperparameters are tuned with stratified 5-fold cross-validation, scored by PR-AUC. In the 90/10 setup the fake class is reweighted (`class_weight` / `scale_pos_weight`).
- **Thresholds:** each model is scored at the default 0.5 and at a "tuned" threshold. The tuned threshold maximizes F1 on out-of-fold training predictions and never looks at the test set.
- **Cross-evaluation:** every model is scored on both the 50/50 and the 90/10 test sets.

Outputs:
- `models/{balanced,imbalanced}_{logreg,rf,xgb}.joblib`: the full pipeline (preprocessing + model).
- `models/*_xgb.json`: native XGBoost weights.
- `models/*.meta.json`: training size, class ratio, best hyperparameters and tuned threshold.
- `results/summary.csv` and `results/metrics_all_seeds.csv`.

### 2. Collect the live sample: `notebooks/02_collect_live_data.ipynb`
- **Sampling:** collects the commenters on the public posts listed in `data/seed_posts.txt` (Apify comment scraper), then fetches their public profiles (Apify profile scraper). The seed posts mix ordinary posts with posts that attract spam (giveaways, promotions), and were not picked by how fake the commenters look.
- **Running the scrapers:** the Apify calls only run with `RUN_COLLECTION = True` and `APIFY_TOKEN` set (free Apify account → Settings → Integrations). By default the notebook rebuilds the feature table from the saved `data/live_raw.jsonl` and checks it against `data/live_labeled.csv`.
- **Default avatar:** "no profile picture" is detected from Instagram's default avatar image id, and that id can change. **Check `profile pic` by eye on a few rows** after a new collection.
- **Privacy:** usernames are never printed.

### 3. Label: see `LABELING_RUBRIC.md`
Two people label every profile independently. Then report Cohen's kappa, settle disagreements, and save the result as `data/live_labeled.csv`.

### 4. Evaluate "in the wild": `notebooks/03_evaluate_live.ipynb`
Scores the labeled live sample with all 6 saved models at both thresholds and computes stratified bootstrap 95% CIs. It also draws PR curves (live vs Kaggle), compares feature distributions (live vs Kaggle) and analyses the false positives. It ends with the findings and limitations. Results are summarised in `results/RESULTS.md`.

## Dashboard: `app.py`
```bash
streamlit run app.py
```
Enter a profile (or raw feature values) to get the risk score from the 50/50 model and the 90/10 model side by side, along with the features that drive each score. A threshold slider is included. The second tab shows precision and recall versus threshold at different fake shares.

## Privacy
Collect only public profile metadata. `data/live_*` is git-ignored. Drop usernames after labeling, and don't publish raw profiles.
