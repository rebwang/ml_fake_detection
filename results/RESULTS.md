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

Labeled live Instagram profiles (commenters on public posts, collected 2026-11-14, relabeled 2026-11-22), scored with the six saved models. Produced by `notebooks/03_evaluate_live.ipynb`, which has the same tables, charts and findings.

<!-- live-auto-start -->

**Live vs Kaggle, default threshold 0.5.** Live: 239 labeled profiles, 14 fake (5.9%). Brackets are stratified bootstrap 95% CIs (1000 resamples). Kaggle columns are the mean over the 10 90/10 test samples.

| Model | Kaggle 90/10 P / R / F1 | Kaggle 90/10 PR-AUC | Live precision | Live recall | Live F1 | Live PR-AUC | Live FP / genuine |
|---|---|---|---|---|---|---|---|
| LogReg (50/50) | 0.49 / 0.97 / 0.65 | 0.89 | 0.13 [0.11–0.15] | 1.00 [1.00–1.00] | 0.23 [0.21–0.26] | 0.76 [0.58–0.94] | 94 / 225 |
| RF (50/50) | 0.61 / 0.90 / 0.73 | 0.93 | 0.17 [0.15–0.20] | 1.00 [1.00–1.00] | 0.29 [0.25–0.33] | 0.64 [0.43–0.86] | 69 / 225 |
| XGBoost (50/50) | 0.58 / 0.97 / 0.72 | 0.95 | 0.14 [0.12–0.16] | 1.00 [1.00–1.00] | 0.25 [0.22–0.28] | 0.77 [0.59–0.93] | 85 / 225 |
| LogReg (90/10) | 0.61 / 0.89 / 0.71 | 0.87 | 0.18 [0.16–0.22] | 1.00 [1.00–1.00] | 0.31 [0.27–0.36] | 0.72 [0.52–0.92] | 62 / 225 |
| RF (90/10) | 0.74 / 0.73 / 0.71 | 0.87 | 0.24 [0.17–0.33] | 0.79 [0.57–1.00] | 0.37 [0.26–0.49] | 0.56 [0.35–0.79] | 34 / 225 |
| XGBoost (90/10) | 0.83 / 0.70 / 0.74 | 0.89 | 0.24 [0.20–0.30] | 1.00 [1.00–1.00] | 0.39 [0.33–0.46] | 0.59 [0.40–0.79] | 44 / 225 |

**Feature shift, live vs Kaggle (Kaggle = train.csv).**

| statistic | Kaggle genuine | live genuine | Kaggle fake | live fake |
|---|---|---|---|---|
| n | 288 | 225 | 288 | 14 |
| median #posts | 74 | 44 | 0 | 0 |
| median #followers | 662 | 229 | 40 | 0.50 |
| median #follows | 431 | 603 | 70 | 568 |
| median follows/followers | 0.90 | 2.99 | 2.28 | 48 |
| median description length | 27 | 17 | 0 | 0 |
| median fullname words | 2 | 2 | 1 | 0.50 |
| % private | 40 | 39 | 37 | 43 |
| % no profile pic | 0.69 | 3.11 | 59 | 64 |
| % external URL | 23 | 15 | 0 | 0 |
| % name==username | 0.35 | 12 | 6.60 | 0 |
| % username has digits | 19 | 39 | 77 | 79 |
| mean digit share username | 0.04 | 0.09 | 0.29 | 0.34 |
| % 0 posts | 1.39 | 14 | 53 | 79 |
| % empty bio | 27 | 42 | 86 | 93 |

**False positives vs the true classes** (FP = genuine accounts flagged at 0.5).

| statistic | FP XGBoost (50/50) | TN XGBoost (50/50) | FP LogReg (90/10) | TN LogReg (90/10) | live genuine (all) | live fake | Kaggle train genuine | Kaggle train fake |
|---|---|---|---|---|---|---|---|---|
| n | 85 | 140 | 62 | 163 | 225 | 14 | 288 | 288 |
| median #posts | 3 | 182 | 1 | 111 | 44 | 0 | 74 | 0 |
| median #followers | 43 | 406 | 40 | 344 | 229 | 0.50 | 662 | 40 |
| median #follows | 271 | 822 | 176 | 894 | 603 | 568 | 431 | 70 |
| % private | 38 | 40 | 19 | 47 | 39 | 43 | 40 | 37 |
| % no profile pic | 8.24 | 0 | 11 | 0 | 3.11 | 64 | 0.69 | 59 |
| % external URL | 2.35 | 23 | 1.61 | 20 | 15 | 0 | 23 | 0 |
| % username has digits | 59 | 26 | 68 | 28 | 39 | 79 | 19 | 77 |
| % empty bio | 67 | 27 | 60 | 36 | 42 | 93 | 27 | 86 |

**Accounts that disappeared after collection.** 1 account(s) marked `gone` (1 labeled, fake). Model scores:

| model | mean_p_fake | share_flagged_0.5 |
|---|---|---|
| LogReg (50/50) | 1.0 | 1.0 |
| RF (50/50) | 0.99 | 1.0 |
| XGBoost (50/50) | 1.0 | 1.0 |
| LogReg (90/10) | 0.99 | 1.0 |
| RF (90/10) | 0.99 | 1.0 |
| XGBoost (90/10) | 1.0 | 1.0 |

Files: `live_results.csv`, `live_ci.csv`, `live_pr_curves.png`, `live_error_analysis.csv`, `live_vs_kaggle_shift.csv`.

<!-- live-auto-end -->

## What this shows
- **The models still rank live accounts well, but the base rate is low.** Live ROC-AUC is 0.93–0.98, close to the Kaggle 90/10 test sets (0.98–0.99). Live PR-AUC is 0.56–0.77, against 0.87–0.95 on Kaggle 90/10. Part of that gap is the lower base rate (5.9% fake vs 10%), which pulls PR-AUC down even for the same ranking quality.
- **Recall survives; precision collapses.** Five of the six models catch all 14 fakes at 0.5 (RF 90/10 catches 11). But the 50/50 models flag 69–94 of the 225 genuine accounts (31–42%), so their precision is only 0.13–0.17. The 90/10 models flag fewer genuine accounts (34–62) and reach precision 0.18–0.24.
- **Training at the realistic ratio helps on live data.** XGBoost (90/10) has the best F1 at 0.5 (0.39, CI 0.33–0.46), with full recall and 44 false positives. Its precision CI (0.20–0.30) and F1 CI do not overlap those of LogReg (50/50) and XGBoost (50/50). XGBoost (50/50) and LogReg (50/50) still have the best PR-AUC (0.77 and 0.76): they rank well, but 0.5 is far too loose a threshold for a 5.9% base rate.
- **Tuned thresholds help a little.** LogReg (90/10) at its tuned 0.68 cuts false positives from 62 to 39 and keeps recall at 1.00 (precision 0.26, F1 0.42, the best of any configuration). XGBoost (90/10) at 0.85 has the fewest false positives (25) but misses 4 of 14 fakes. No tuned threshold gets precision above 0.29.
- **The cause is domain shift in the genuine class.** Live genuine accounts are much less "active" than Kaggle genuine accounts: median followers 229 vs 662, follows/followers 3.0 vs 0.9, 14% with zero posts vs 1.4%, 39% with digits in the username vs 19%, and 42% with an empty bio vs 27%. Each of these moves them toward what the models learned as "fake".
- **The false positives are low-activity genuine users.** The genuine accounts XGBoost (50/50) flags have a median of 3 posts and 43 followers. 32% have zero posts, 59% have digits in the username and 67% have an empty bio, which is close to the Kaggle fake profile. The LogReg (90/10) false positives look the same (median 1 post, 40 followers, 40% zero posts). The 11 features cannot tell a lurker or new real user from a spam account.
- **Live fakes look like Kaggle fakes, except that they follow many accounts.** 64% lack a profile picture (59% in Kaggle), 79% have zero posts (53%) and 93% have an empty bio (86%), so they are empty shells that the models catch easily. The difference is follows: median 568 vs 70, with a median of 0.5 followers.
- **Takeaway:** the fake side of Kaggle transfers to live data; the genuine side does not. Training on 90/10 data and raising the threshold reduce false positives, but at a 5.9% base rate even the best configuration flags about 2.8 genuine accounts per fake caught (39 FP vs 14 TP). A usable model would need genuine training data from the target population, or richer features (post content, comment behaviour, account age).

## Limitations
- **One labeler.** All labels are Rebecca's judgment from public profiles, so there is no inter-annotator agreement (no Cohen's kappa). Low-activity accounts, exactly where the models disagree with the labels, are the hardest to judge.
- **Full coverage.** All 239 collected profiles are labeled, including the 18 that an earlier pass left blank as too uncertain. No hard cases are dropped, but those labels are the least certain.
- **Sampling.** Profiles are commenters on a handful of seed posts (`data/seed_posts.txt`), so the sample reflects those posts' audiences, not Instagram overall. Not every seed post contributes equally.
- **Private accounts.** About 39% are private, so the labeler saw only the profile header.
- **Few fakes.** Only 14 fakes, so recall and PR-AUC CIs are wide (e.g. XGBoost 50/50 PR-AUC 0.59–0.93), and catching all 14 is consistent with a true recall as low as about 0.8. One of the 14 is the `gone` account with an empty record. Small differences between models are not meaningful.
