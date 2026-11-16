"""Collect live public Instagram profiles and turn them into the 11 dataset features.

Input: a text file with one username per line (e.g. commenters on a few public
posts via --comments-of, or followers gathered by hand or with --followers-of).

  # Step 1 (recommended): build the username list from commenters on public posts.
  # data/seed_posts.txt: 3-5 public post/reel URLs, one per line (mix ordinary posts
  # and giveaway/promo posts; don't pick posts by how fake the commenters look).
  export APIFY_TOKEN=...
  python src/collect_live.py --comments-of data/seed_posts.txt --per-post 80

  # Apify (recommended): runs on Apify's servers, so no risk to your own IP
  export APIFY_TOKEN=...            # from https://console.apify.com/account/integrations
  python src/collect_live.py --source apify --usernames data/live_usernames.csv


  # Instaloader (backup): keep the sample small; logging in is usually required now
  python src/collect_live.py --source instaloader --usernames data/live_usernames.csv --login YOUR_THROWAWAY

  # Optional: build the username list from an account's followers (Instaloader, needs login)
  python src/collect_live.py --source instaloader --followers-of some_account --max 300 --login YOUR_THROWAWAY

Output:
  data/live_raw.jsonl        raw metadata as returned (keep private; do not commit)
  data/live_to_label.csv     features + empty labeler_a / labeler_b columns for manual labeling
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests

from features import FEATURES, profile_to_features

# Instagram's default "no photo" avatar. Profiles whose picture URL contains this
# id are treated as having no profile picture. Check a few by eye before trusting it.
DEFAULT_AVATAR_MARKERS = [
    "44884218_345707102882519_2446069589734326272_n",  # older default avatar
    "573323465_1219825463302212_7278921664109726296_n",  # default avatar seen Nov 2026
]
APIFY_ACTOR = "apify~instagram-profile-scraper"
APIFY_COMMENTS_ACTOR = "apify~instagram-comment-scraper"


def _apify_token():
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        raise SystemExit("set the APIFY_TOKEN environment variable first")
    return token


def has_pic(url: str) -> bool:
    return bool(url) and not any(m in url for m in DEFAULT_AVATAR_MARKERS)


def from_apify(usernames, token):
    url = f"https://api.apify.com/v2/acts/{APIFY_ACTOR}/run-sync-get-dataset-items"
    out = []
    for i in range(0, len(usernames), 50):
        batch = usernames[i:i + 50]
        r = requests.post(url, params={"token": token}, json={"usernames": batch}, timeout=600)
        r.raise_for_status()
        for p in r.json():
            if not p.get("username"):
                continue
            out.append({
                "raw": p,
                "features": profile_to_features(
                    p.get("username"), p.get("fullName"), p.get("biography"),
                    p.get("externalUrl"), p.get("private"), p.get("postsCount"),
                    p.get("followersCount"), p.get("followsCount"),
                    has_pic(p.get("profilePicUrl") or p.get("profilePicUrlHD"))),
                "username": p["username"],
            })
        print(f"apify: {len(out)} profiles so far")
    return out


def commenters_of(post_urls, per_post, token):
    """Usernames of people who commented on the given public posts/reels (Apify)."""
    url = f"https://api.apify.com/v2/acts/{APIFY_COMMENTS_ACTOR}/run-sync-get-dataset-items"
    r = requests.post(url, params={"token": token},
                      json={"directUrls": post_urls, "resultsLimit": per_post}, timeout=600)
    r.raise_for_status()
    return [c["ownerUsername"] for c in r.json() if c.get("ownerUsername")]


def _loader(login):
    import instaloader
    L = instaloader.Instaloader(download_pictures=False, download_videos=False,
                                save_metadata=False, quiet=True)
    if login:
        try:
            L.load_session_from_file(login)
        except FileNotFoundError:
            L.interactive_login(login)
            L.save_session_to_file()
    return L


def from_instaloader(usernames, login):
    import instaloader
    L = _loader(login)
    out = []
    for u in usernames:
        try:
            p = instaloader.Profile.from_username(L.context, u)
        except Exception as e:  # deleted, renamed, or rate-limited
            print(f"skip {u}: {e}")
            continue
        out.append({
            "raw": {"username": p.username, "full_name": p.full_name, "biography": p.biography,
                    "external_url": p.external_url, "is_private": p.is_private,
                    "mediacount": p.mediacount, "followers": p.followers,
                    "followees": p.followees, "profile_pic_url": p.profile_pic_url},
            "features": profile_to_features(
                p.username, p.full_name, p.biography, p.external_url, p.is_private,
                p.mediacount, p.followers, p.followees, has_pic(p.profile_pic_url)),
            "username": p.username,
        })
        time.sleep(random.uniform(4, 9))  # slow pacing to avoid rate limits
    return out


def followers_of(account, n, login):
    import instaloader
    L = _loader(login)
    prof = instaloader.Profile.from_username(L.context, account)
    names = []
    for f in prof.get_followers():
        names.append(f.username)
        if len(names) >= n:
            break
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["apify", "instaloader"], default="apify")
    ap.add_argument("--usernames", help="text file, one username per line")
    ap.add_argument("--followers-of", help="collect follower usernames of this account (Instaloader)")
    ap.add_argument("--comments-of", help="text file of public post/reel URLs; collect commenter usernames (Apify)")
    ap.add_argument("--per-post", type=int, default=80, help="max comments per post for --comments-of")
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--login", help="Instagram username for Instaloader (throwaway account)")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.comments_of or a.followers_of:
        if a.comments_of:
            posts = [l.strip() for l in open(a.comments_of) if l.strip() and not l.startswith("#")]
            names = commenters_of(posts, a.per_post, _apify_token())
        else:
            names = followers_of(a.followers_of, a.max, a.login)
        path = out / "live_usernames.csv"
        existing = [l.strip().lstrip("@") for l in open(path) if l.strip()] if path.exists() else []
        merged = list(dict.fromkeys(existing + [n.lstrip("@") for n in names]))
        path.write_text("\n".join(merged) + "\n")
        print(f"added {len(merged) - len(existing)} new usernames to {path} (total {len(merged)})")
        return

    names = [l.strip().lstrip("@") for l in open(a.usernames) if l.strip()]
    names = list(dict.fromkeys(names))[:a.max]
    if a.source == "apify":
        recs = from_apify(names, _apify_token())
    else:
        recs = from_instaloader(names, a.login)

    with open(out / "live_raw.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r["raw"], default=str) + "\n")
    df = pd.DataFrame([{"username": r["username"], **r["features"]} for r in recs])
    df["profile_url"] = "https://www.instagram.com/" + df["username"] + "/"
    df["labeler_a"] = ""   # 1 = fake, 0 = genuine, blank = unsure
    df["labeler_b"] = ""
    df["fake"] = ""        # final agreed label, filled after both have labeled
    df["collected_on"] = pd.Timestamp.today().date().isoformat()
    df[["username", "profile_url"] + FEATURES + ["labeler_a", "labeler_b", "fake", "collected_on"]] \
        .to_csv(out / "live_to_label.csv", index=False)
    print(f"wrote {len(df)} profiles to {out / 'live_to_label.csv'}")


if __name__ == "__main__":
    main()
