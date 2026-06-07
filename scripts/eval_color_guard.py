"""Calibrate the torso color guard on held-out labeled crops (no overfit).

Measures same-person vs different-person torso color-histogram similarity, split
by within-camera vs cross-camera, to (a) confirm the color signal separates
distinct people WITHIN a camera and (b) pick a veto threshold that almost never
rejects a true same-person. Mirrors the embedding-eval discipline.

Torso HS histogram: crop region height[0.15,0.55] x width[0.15,0.85], BGR->HSV,
mask V>40 (drop very dark), 2D hist H(16)xS(4) so both hue and gray/colorful are
captured. compareHist CORREL (higher = more similar color).

Usage: python scripts/eval_color_guard.py /tmp/labeled_5152/label5152 /tmp/labeled_crops/label
"""
import sys, os, glob, re
import numpy as np
import cv2

H_BINS, S_BINS = 16, 4
TORSO = (0.15, 0.55, 0.15, 0.85)  # y0,y1,x0,x1 ratios
V_MIN = 40

def torso_hist(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    y0, y1, x0, x1 = TORSO
    roi = img[int(h*y0):int(h*y1), int(w*x0):int(w*x1)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] > V_MIN).astype(np.uint8)
    if mask.sum() < 20:
        return None
    hist = cv2.calcHist([hsv], [0, 1], mask, [H_BINS, S_BINS], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def device_of(fname):
    m = re.search(r"device_?(\d+)", os.path.basename(fname))
    return m.group(1) if m else "?"

def load(label_dir):
    items = []  # (person, device, hist)
    for person in sorted(os.listdir(label_dir)):
        pdir = os.path.join(label_dir, person)
        if not os.path.isdir(pdir):
            continue
        for p in glob.glob(os.path.join(pdir, "*")):
            if not p.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            hh = torso_hist(p)
            if hh is not None:
                items.append((f"{label_dir}:{person}", device_of(p), hh))
    return items

def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")

all_items = []
for d in sys.argv[1:]:
    its = load(d)
    print(f"{d}: {len(its)} crops, persons={len(set(i[0] for i in its))}")
    all_items += its

# AGGREGATE regime (matches the guard: tracklet-mean vs person-mean). Group crops
# by (set, person, device); split into ~CHUNK-sized pseudo-tracklets; aggregate
# each chunk into one mean histogram; compare chunk-vs-chunk.
CHUNK = 10
def agg(hists):
    m = np.mean(np.stack(hists), axis=0).astype(np.float32)
    cv2.normalize(m, m, 0, 1, cv2.NORM_MINMAX)
    return m

groups = {}  # (set,person,device) -> [hist...]
for setname_p, dev, hh in [(it[0], it[1], it[2]) for it in all_items]:
    groups.setdefault((setname_p.split(":")[0], setname_p, dev), []).append(hh)

# Build pseudo-tracklet descriptors per group.
chunks = {}  # key -> [agg_desc...]
for key, hs in groups.items():
    if len(hs) < 4:
        continue
    descs = [agg(hs[i:i+CHUNK]) for i in range(0, len(hs), CHUNK) if len(hs[i:i+CHUNK]) >= 3]
    if descs:
        chunks[key] = descs

cats = {"same_within": [], "same_cross": [], "diff_within": [], "diff_cross": []}
keys = list(chunks)
for a in range(len(keys)):
    sa, pa, da = keys[a]
    for da_i in range(len(chunks[keys[a]])):
        for b in range(len(keys)):
            sb, pb, db = keys[b]
            if sa != sb:
                continue  # don't cross datasets
            for db_i in range(len(chunks[keys[b]])):
                if a == b and db_i <= da_i:
                    continue
                sim = float(cv2.compareHist(chunks[keys[a]][da_i], chunks[keys[b]][db_i], cv2.HISTCMP_CORREL))
                same = pa == pb
                within = da == db
                cats[("same_" if same else "diff_") + ("within" if within else "cross")].append(sim)

print("\n=== torso color_sim (CORREL, higher=more similar) ===")
for k, v in cats.items():
    if v:
        print(f"{k:12} n={len(v):6} mean={np.mean(v):.3f} p05={pct(v,5):.3f} "
              f"p25={pct(v,25):.3f} p50={pct(v,50):.3f} p75={pct(v,75):.3f} p95={pct(v,95):.3f}")

sw, dw = cats["same_within"], cats["diff_within"]
if sw and dw:
    # Veto if color_sim < threshold. Want: almost never veto same (threshold below
    # same_within low pct) while catching many diff. Report tradeoff at same_p05/p02.
    for q in (2, 5, 10):
        thr = pct(sw, q)
        false_veto = np.mean(np.array(sw) < thr) * 100
        caught = np.mean(np.array(dw) < thr) * 100
        print(f"\nthr=same_within_p{q:02d}={thr:.3f}: "
              f"false-veto same-within={false_veto:.1f}%  catches diff-within={caught:.1f}%")
    print("\n=> pick threshold near same_within_p02..p05 (minimize false-veto). "
          "diff-within caught% = upper bound of how many wrong-color attaches we can block.")
