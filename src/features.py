"""Feature definitions shared by training, live collection and the dashboard.

Column names match the Kaggle "Instagram Fake Spammer Genuine Accounts" files
(train.csv / test.csv). The label column is `fake` (1 = fake, 0 = genuine).
"""
import re

FEATURES = [
    "profile pic",           # 1 if the account has a (non-default) profile picture
    "nums/length username",  # share of digits in the username
    "fullname words",        # number of words in the full name
    "nums/length fullname",  # share of digits in the full name
    "name==username",        # 1 if full name equals username
    "description length",    # characters in the bio
    "external URL",          # 1 if the profile has an external link
    "private",               # 1 if the account is private
    "#posts",
    "#followers",
    "#follows",
]
LABEL = "fake"
COUNT_FEATURES = ["description length", "#posts", "#followers", "#follows"]


def _digit_ratio(s: str) -> float:
    s = s or ""
    return round(sum(c.isdigit() for c in s) / len(s), 2) if s else 0.0


def profile_to_features(username, full_name, biography, external_url,
                        is_private, posts, followers, follows,
                        has_profile_pic) -> dict:
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
