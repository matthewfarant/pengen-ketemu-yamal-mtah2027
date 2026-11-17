"""Find each half's kickoff, so everything before it in the graded window is a dead ball.

The game has not started before the kickoff, and the external training matches begin at their
kickoffs, so the classifier has never seen a team walking out for the second half. The Smart
Tagging (an automatic input) marks a window of a few seconds around every "Kick Off (Start)"; the
tracking pins the instant inside it: the first frame where the ball, resting near the centre spot,
starts to move."""
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25


def kickoff_tags(xml_path):
    out = []
    for inst in ET.parse(xml_path).iter("instance"):
        labels = {l.findtext("group"): l.findtext("text") for l in inst.findall("label")}
        if (inst.findtext("code") or "").strip().lower() == "set pieces" and \
                (labels.get("Type") or "").strip().lower() == "kick off (start)":
            s = int(round(float(inst.findtext("start")) * FPS))
            e = int(round(float(inst.findtext("end")) * FPS))
            out.append((s, e))
    return sorted(out)


def kickoff_frame(match, start, end, margin=50):
    """First frame in [start - margin, end + margin) where the ball is within 3 m of the centre
    spot and has moved more than 1.5 m half a second later. Falls back to the window's middle."""
    cx, cy = match.length / 2, match.width / 2
    a, b = max(start - margin, 0), min(end + margin, match.n - 13)
    for t in range(a, b):
        if not np.isfinite(match.bx[t]) or not np.isfinite(match.bx[t + 12]):
            continue
        near = np.hypot(match.bx[t] - cx, match.by[t] - cy) < 3
        moved = np.hypot(match.bx[t + 12] - match.bx[t], match.by[t + 12] - match.by[t]) > 1.5
        if near and moved:
            # step back to the last frame the ball was still
            while t > a and np.isfinite(match.bx[t - 1]) and \
                    np.hypot(match.bx[t] - match.bx[t - 1], match.by[t] - match.by[t - 1]) > 0.05:
                t -= 1
            return t
    return (start + end) // 2


def kickoffs_per_block(match, xml_path, blocks):
    """For each graded block (start, stop): the kickoff frame of the first kickoff tag that starts
    inside or just before the block, or the block start if none."""
    tags = kickoff_tags(xml_path)
    out = []
    for a, b in blocks:
        cand = [(s, e) for s, e in tags if a - 10 * FPS <= s < b]
        out.append(kickoff_frame(match, *cand[0]) if cand else a)
    return out
