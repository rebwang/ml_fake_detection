# ml_fake_detection

Fake account detection: CS6140 project code.

This code covers the professor's feedback. It trains models on **50/50** data and on **90/10** data, saves the weights for each, and tests both sets of models side by side on a **live, uncurated** sample of Instagram profiles.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Put the Kaggle files `train.csv` and `test.csv` (Instagram Fake Spammer Genuine Accounts) in `data/`.

## 1. Train and save models: `src/train.py`
```bash
python src/train.py            # --seeds 10 --fake-share 0.10 are the defaults
```
- **Balanced (50/50):** trains on Kaggle's train.csv and tests on test.csv. Exact duplicate rows are dropped from train.csv first.
- **Imbalanced (90/10):** keeps every genuine account and randomly downsamples fake accounts to 10%, in both train and test. Train and test are built from Kaggle's own split, so they never overlap. There are only about 32 fake accounts in training and 7 in test, so this is repeated over 10 random seeds and the mean ± std is reported. The seed-0 models are the ones saved.
- **Models:** Logistic Regression, Random Forest and XGBoost. Hyperparameters are tuned with stratified 5-fold cross-validation, scored by PR-AUC. In the 90/10 setup the fake class is reweighted (`class_weight` / `scale_pos_weight`).
- **Thresholds:** each model is scored at the default 0.5 and at a "tuned" threshold. The tuned threshold maximizes F1 on out-of-fold training predictions and never looks at the test set.
- **Cross-evaluation:** every model is scored on both the 50/50 and the 90/10 test sets. This shows what happens when a model trained on balanced data meets realistic class ratios.

Outputs:
- `models/{balanced,imbalanced}_{logreg,rf,xgb}.joblib`: the full pipeline (preprocessing + model).
- `models/*_xgb.json`: native XGBoost weights.
- `models/*.meta.json`: training size, class ratio, best hyperparameters and tuned threshold.
- `results/summary.csv` and `results/metrics_all_seeds.csv`.

## 2. Collect the live sample: `src/collect_live.py`
1. Build `data/live_usernames.txt` (about 200 to 300 usernames) from the commenters on a few public posts. Create `data/seed_posts.txt` with 3 to 5 public post or reel URLs, one per line. Mix ordinary posts with posts that attract spam (giveaways, promotions). Don't pick posts by how fake the commenters look, or the sample is no longer uncurated. Then run:
   ```bash
   export APIFY_TOKEN=...   # free Apify account → Settings → Integrations
   python src/collect_live.py --comments-of data/seed_posts.txt --per-post 80
   ```
   This uses Apify's official comment scraper. It merges the commenters into `data/live_usernames.txt` without duplicates, so you can run it again with more posts. You can also add usernames to the file by hand.
2. Fetch the profiles:
   ```bash
   export APIFY_TOKEN=...   # free Apify account → Settings → Integrations
   python src/collect_live.py --source apify --usernames data/live_usernames.txt
   ```
   Backup: `--source instaloader --login <throwaway account>`. It is slow on purpose; keep the sample small.
3. This writes `data/live_to_label.csv`, which has the same 11 features as Kaggle plus empty label columns.

**Check `profile pic` by eye on a few rows.** The script detects Instagram's default avatar from its image id, and that id can change.

## 3. Label: see `LABELING_RUBRIC.md`
Two people label every profile independently. Then report Cohen's kappa, settle disagreements, and save the result as `data/live_labeled.csv`.

## 4. Side-by-side "in the wild" evaluation: `src/evaluate_live.py`
```bash
python src/evaluate_live.py --live data/live_labeled.csv
```
This scores the live sample with all 6 saved models, at both thresholds, and writes `results/live_results.csv`.

## 5. Dashboard: `app.py`
```bash
streamlit run app.py
```
Enter a profile (or raw feature values) to get the risk score from the 50/50 model and the 90/10 model side by side, along with the features that drive each score. A threshold slider is included. The second tab shows precision and recall versus threshold at different fake shares.

## Privacy
Collect only public profile metadata. `data/live_*` is git-ignored. Drop usernames after labeling, and don't publish raw profiles.
