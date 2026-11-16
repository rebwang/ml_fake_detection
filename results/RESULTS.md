# First results on the Kaggle data (tuned thresholds)

Metrics are for the fake class. 90/10 test numbers are mean ± std over 10 random samples. Each 90/10 test set has 60 genuine and 7 fake accounts, so the std is large.

| Model | Trained on | Test 50/50: Prec / Rec / F1 | Test 90/10: Prec / Rec / F1 | PR-AUC 50/50 → 90/10 |
|---|---|---|---|---|
| LogReg | 50/50 | 0.88 / 0.93 / 0.90 | 0.46 / 0.97 / 0.62 ± .03 | 0.98 → 0.89 |
| LogReg | 90/10 | 0.97 / 0.78 / 0.87 | 0.78 / 0.73 / 0.75 ± .15 | 0.97 → 0.87 |
| RF | 50/50 | 0.87 / 0.92 / 0.89 | 0.46 / 0.96 / 0.62 ± .05 | 0.98 → 0.93 |
| RF | 90/10 | 0.98 / 0.74 / 0.84 | 0.85 / 0.66 / 0.70 ± .14 | 0.98 → 0.87 |
| **XGBoost** | **50/50** | **0.97 / 0.92 / 0.94** | **0.77 / 0.94 / 0.84 ± .06** | **0.99 → 0.95** |
| XGBoost | 90/10 | 0.99 / 0.73 / 0.83 | 0.90 / 0.66 / 0.74 ± .08 | 0.98 → 0.89 |

## What this shows
- **Models trained on 50/50 data over-flag at realistic ratios.** On a 90/10 test set, about half of LogReg's and RF's fake flags are actually genuine accounts (precision ≈ 0.46). Recall stays high.
- **Models trained on 90/10 data have the opposite problem.** They are more conservative: precision is higher, but they miss about a third of the fakes. They only had 32 fake training examples, and their results vary a lot from sample to sample.
- **XGBoost trained on 50/50 data is the best overall,** with F1 0.84 on 90/10 test data and the highest PR-AUC. Its tuned threshold (0.62) is stricter than 0.5, which offsets the over-flagging.
- **Takeaway for the report:** the class ratio mostly shifts *where the threshold should sit*, rather than how well the models rank accounts. PR-AUC stays fairly high for every model. The live sample will show which setup holds up on real data.

Note: train.csv has 2 exact duplicate rows, which are dropped before training. test.csv also has 2, which are kept so the official test set stays intact.
Full numbers: `summary.csv` (default 0.5 and tuned thresholds) and `metrics_all_seeds.csv`.

# Live sample results

Labeled live Instagram profiles (commenters on public posts, collected 2026-11-14), scored with the six saved models. Produced by `notebooks/03_evaluate_live.ipynb`, which has the same tables, charts and findings.

<!-- live-auto-start -->

**Live vs Kaggle, default threshold 0.5.** Live: 221 labeled profiles, 26 fake (11.8%). Brackets are stratified bootstrap 95% CIs (1000 resamples). Kaggle columns are the mean over the 10 90/10 test samples.

| Model | Kaggle 90/10 P / R / F1 | Kaggle 90/10 PR-AUC | Live precision | Live recall | Live F1 | Live PR-AUC | Live FP / genuine |
|---|---|---|---|---|---|---|---|
| LogReg (50/50) | 0.49 / 0.97 / 0.65 | 0.89 | 0.24 [0.20–0.29] | 0.88 [0.77–1.00] | 0.37 [0.32–0.44] | 0.50 [0.36–0.70] | 74 / 195 |
| RF (50/50) | 0.61 / 0.90 / 0.73 | 0.93 | 0.30 [0.24–0.36] | 0.85 [0.73–0.96] | 0.44 [0.37–0.52] | 0.44 [0.31–0.62] | 52 / 195 |
| XGBoost (50/50) | 0.58 / 0.97 / 0.72 | 0.95 | 0.28 [0.23–0.33] | 0.92 [0.81–1.00] | 0.42 [0.36–0.50] | 0.57 [0.43–0.74] | 63 / 195 |
| LogReg (90/10) | 0.61 / 0.89 / 0.71 | 0.87 | 0.32 [0.27–0.40] | 0.85 [0.73–0.96] | 0.47 [0.39–0.56] | 0.46 [0.32–0.64] | 46 / 195 |
| RF (90/10) | 0.74 / 0.73 / 0.71 | 0.87 | 0.33 [0.22–0.47] | 0.46 [0.27–0.65] | 0.39 [0.25–0.52] | 0.39 [0.27–0.56] | 24 / 195 |
| XGBoost (90/10) | 0.83 / 0.70 / 0.74 | 0.89 | 0.33 [0.24–0.44] | 0.62 [0.42–0.81] | 0.43 [0.32–0.55] | 0.42 [0.30–0.59] | 33 / 195 |

**Feature shift, live vs Kaggle (Kaggle = train.csv).**

| statistic | Kaggle genuine | live genuine | Kaggle fake | live fake |
|---|---|---|---|---|
| n | 288 | 195 | 288 | 26 |
| median #posts | 74 | 58 | 0 | 0.50 |
| median #followers | 662 | 270 | 40 | 66 |
| median #follows | 431 | 603 | 70 | 982 |
| median follows/followers | 0.90 | 2.31 | 2.28 | 19 |
| % private | 40 | 37 | 37 | 35 |
| % no profile pic | 0.69 | 2.56 | 59 | 23 |
| % external URL | 23 | 17 | 0 | 0 |
| % username has digits | 19 | 37 | 77 | 69 |
| mean digit share username | 0.04 | 0.09 | 0.29 | 0.23 |
| median description length | 27 | 24 | 0 | 0 |
| % 0 posts | 1.39 | 11 | 53 | 50 |

**False positives vs the true classes** (FP = genuine accounts flagged at 0.5).

| statistic | FP balanced_xgb | FP imbalanced_logreg | live genuine (all) | live fake | Kaggle train genuine | Kaggle train fake |
|---|---|---|---|---|---|---|
| n | 63 | 46 | 195 | 26 | 288 | 288 |
| median #posts | 4 | 1 | 58 | 0.50 | 74 | 0 |
| median #followers | 43 | 40 | 270 | 66 | 662 | 40 |
| median #follows | 180 | 138 | 603 | 982 | 431 | 70 |
| % private | 35 | 17 | 37 | 35 | 40 | 37 |
| % no profile pic | 7.94 | 11 | 2.56 | 23 | 0.69 | 59 |
| % external URL | 3.17 | 2.17 | 17 | 0 | 23 | 0 |
| % username has digits | 62 | 70 | 37 | 69 | 19 | 77 |
| % empty bio | 63 | 54 | 38 | 85 | 27 | 86 |

**Accounts that disappeared after collection.** 1 account(s) marked `gone` (0 labeled). Model scores:

| model | mean_p_fake | share_flagged_0.5 |
|---|---|---|
| LogReg (50/50) | 1.0 | 1.0 |
| RF (50/50) | 0.99 | 1.0 |
| XGBoost (50/50) | 1.0 | 1.0 |
| LogReg (90/10) | 0.99 | 1.0 |
| RF (90/10) | 0.99 | 1.0 |
| XGBoost (90/10) | 1.0 | 1.0 |

Files: `live_ci.csv`, `live_pr_curves.png`, `live_error_analysis.csv`, `live_vs_kaggle_shift.csv`.

<!-- live-auto-end -->

## What this shows
- **Every model ranks live accounts far worse than Kaggle accounts.** Live PR-AUC is 0.39–0.57 against 0.87–0.95 on the Kaggle 90/10 test sets. ROC-AUC stays around 0.84–0.89, so the models still separate the classes somewhat, but not well enough at an 11.8% base rate.
- **Precision collapses; recall mostly survives.** The 50/50 models catch 85–92% of fakes but flag 52–74 of the 195 genuine accounts (27–38%), so precision is only 0.24–0.30. The 90/10 models flag fewer genuine accounts (24–46) but RF and XGBoost then miss 38–54% of fakes. Precision CIs overlap across all six models, so no model is clearly better on live data. XGBoost (50/50) has the best point PR-AUC (0.57), as on Kaggle.
- **Thresholds tuned on Kaggle don't help on live data.** LogReg (90/10) at its tuned 0.68 cuts false positives from 46 to 28 but recall falls from 0.85 to 0.65; XGBoost (90/10) at 0.85 has the fewest false positives (18) but misses 15 of 26 fakes. No tuned threshold gets precision above 0.38 (`live_results.csv`).
- **The cause is domain shift in the genuine class.** Live genuine accounts are much less "active" than Kaggle genuine accounts: median followers 270 vs 662, follows/followers 2.3 vs 0.9, 11% with zero posts vs 1.4%, 37% with digits in the username vs 19%, and 38% with an empty bio vs 27%. Each of these moves them toward what the models learned as "fake".
- **The false positives are exactly those low-activity genuine accounts.** The genuine accounts that XGBoost (50/50) flags have a median of 4 posts and 43 followers, 30% have zero posts, 62% have digits in the username, and 63% have an empty bio. That profile is close to the Kaggle fake class (median 0 posts, 40 followers, 77% digits, 86% empty bio). The features cannot tell a lurker or new real user from a spam account. The LogReg (90/10) false positives look the same.
- **Live fakes also look different from Kaggle fakes.** Only 23% of live fakes lack a profile picture (59% in Kaggle), and they follow far more accounts (median 982 vs 70). Kaggle's fakes are mostly empty shells; the live fakes are follow-spam accounts with a picture. The models lean on "no profile picture", which is less informative here.
- **Takeaway for the report:** the Kaggle results do not transfer. The class ratio was not the main problem; the feature distributions of both classes moved. A deployable model would need training data from the target population, or features beyond these 11 (e.g. post content, comment behaviour, account age).

## Limitations
- **One labeler.** Every label is Rebecca's; there is no second labeler, so no Cohen's kappa. Labels from public profiles are a judgment call, especially for low-activity accounts, which is exactly where the models disagree with the labels.
- **Incomplete coverage.** 221 of 239 collected profiles are labeled; 18 were left blank as too uncertain (12 of the 18 are private, median 14 posts and 53 followers). Dropping the hardest cases likely makes the live numbers look somewhat better than they would be on the full sample.
- **Sampling.** Profiles are commenters on a handful of seed posts (`data/seed_posts.txt`), so the sample reflects those posts' audiences, not Instagram overall. Not every seed post contributes equally.
- **Private accounts.** About 37% of the labeled profiles are private, so the labeler saw only the profile header for them.
- **Few fakes.** Only 26 fakes, so recall and PR-AUC CIs are wide (e.g. XGBoost 50/50 PR-AUC 0.43–0.74). Small differences between models are not meaningful.
- **Deleted accounts.** Only one account has been marked `gone` so far, and it was unlabeled with all-zero features (the scrape returned nothing for it), so its model scores carry no information. The planned re-check of fake-labeled accounts 2–3 weeks after collection is still to do.
