import os, numpy as np, glob
from sklearn.model_selection import train_test_split

INPUT_DIR, OUTPUT_DIR = "blocks_dataset", "final_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npz")))
blocks, majors = [], []

for f in files:
    try:
        d = np.load(f)
        p, l = d['points'], d['labels']
        if len(p) == 0 or len(l) == 0: continue
        blocks.append((os.path.basename(f), p, l, d.get('city', 'unknown').item(), d.get('origin', None)))
        majors.append(np.bincount(l).argmax())
    except: continue

try:
    idx = np.arange(len(blocks))
    tr, vt = train_test_split(idx, test_size=0.3, stratify=majors, random_state=42)
    va, te = train_test_split(vt, test_size=0.5, stratify=np.array(majors)[vt], random_state=42)
except:
    idx = np.random.permutation(len(blocks))
    tr, va, te = np.split(idx, [int(.7*len(idx)), int(.85*len(idx))])

def save(name, indices):
    path = os.path.join(OUTPUT_DIR, name)
    os.makedirs(path, exist_ok=True)
    for i in indices:
        fn, pts, lbl, city, origin = blocks[i]
        np.savez_compressed(os.path.join(path, fn), points=pts, labels=lbl, city=city, origin=origin)

save("train", tr); save("val", va); save("test", te)
print(f"Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}")