# Labeling rubric for the live sample

Two group members label every profile independently in `data/live_to_label.csv`, without discussing it first.
Put **1** in your column if the account is fake or spam, **0** if it is genuine, and leave it blank if you're unsure.
Open the `profile_url` for each profile and look at the actual posts, not only the numbers.

**Signs of a fake or spam account (two or more strong signs → 1):**
- Default avatar, or a stock or stolen-looking photo
- No posts, or only reposted promo content
- Follows many accounts (hundreds or thousands) but has very few followers
- Username looks auto-generated (random letters followed by a run of digits)
- Bio pushes crypto, giveaways, "DM for promo", adult content, or a link shortener
- Comments are generic ("Nice pic! 🔥") or the account only interacts with promotional posts

**Signs of a genuine account:**
- Personal photos taken over time, with captions and real interaction (replies from friends)
- A balanced follower/following ratio, or the profile of a real business or creator

**Private accounts:** label them from what you can see, and leave the label blank if there isn't enough to judge.

**After labeling:**
1. Compute Cohen's kappa between the two labelers on the profiles both labeled, e.g. `sklearn.metrics.cohen_kappa_score(df.labeler_a, df.labeler_b)`.
2. Settle every disagreement together and write the final label in the `fake` column. Drop profiles that are still unsure.
3. About 2 to 3 weeks later, recheck the profiles labeled fake. An account that has since been deleted or suspended is extra evidence that it was fake. Report how many.
4. Save the file as `data/live_labeled.csv` and remove usernames from anything you share.
