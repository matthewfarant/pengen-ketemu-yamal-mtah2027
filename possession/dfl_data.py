"""Download (figshare, CC BY 4.0) and cache the Sportec Open DFL matches."""
import glob
import json
import os
import subprocess
import urllib.error
import urllib.request

from . import io_dfl
from .common import Match
from .labels import apply_control_rule

ARTICLE = "https://api.figshare.com/v2/articles/28196177"
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def _fetch(url, path=None):
    """urllib first; curl as a fallback for Pythons without a CA bundle (python.org macOS)."""
    try:
        if path:
            urllib.request.urlretrieve(url, path)
            return None
        with urllib.request.urlopen(url) as r:
            return r.read()
    except urllib.error.URLError:
        cmd = ["curl", "-sSL", url] + (["-o", path] if path else [])
        out = subprocess.run(cmd, check=True, capture_output=True).stdout
        return None if path else out


def have_all(dest):
    return all(glob.glob(os.path.join(dest, f"*{kind}*{m}.xml"))
               for m in MATCHES for kind in ("matchinformation", "positions"))


def download(dest):
    os.makedirs(dest, exist_ok=True)
    files = json.loads(_fetch(ARTICLE))["files"]
    for f in files:
        if "events" in f["name"]:
            continue
        path = os.path.join(dest, f["name"])
        if not os.path.exists(path) or os.path.getsize(path) != f["size"]:
            print("downloading", f["name"])
            _fetch(f["download_url"], path)


def load(mid, raw_dir, cache_dir):
    path = os.path.join(cache_dir, f"dfl_{mid}.npz")
    if not os.path.exists(path):
        info = glob.glob(os.path.join(raw_dir, f"*matchinformation*{mid}.xml"))[0]
        pos = glob.glob(os.path.join(raw_dir, f"*positions*{mid}.xml"))[0]
        io_dfl.load(info, pos, mid).save(path)
    m = Match.load(path)
    m.labels = apply_control_rule(m.labels, m.half)
    return m
