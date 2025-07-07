import os, numpy as np, glob
from sklearn.model_selection import train_test_split

INPUT_DIR, OUTPUT_DIR = "blocks_dataset", "final_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.npz")))
blocks, labels_major = [], []

for f in files:
    try:
        data = np.load(f)
        p, l = data['points'], data['labels']
        if len(p) != len(l): continue
        blocks.append((os.path.basename(f), p, l, data.get('city', 'unknown').item(), data.get('origin', None)))
        u, c = np.unique(l, return_counts=True)
        labels_major.append(u[np.argmax(c)])
    except: continue

try:
    idx = np.arange(len(blocks))
    tr, valtest = train_test_split(idx, test_size=0.3, stratify=labels_major, random_state=42)
    va, te = train_test_split(valtest, test_size=0.5, stratify=np.array(labels_major)[valtest], random_state=42)
except:
    idx = np.random.permutation(len(blocks))
    tr, va, te = np.split(idx, [int(.7*len(idx)), int(.85*len(idx))])

def save(name, indices):
    path = os.path.join(OUTPUT_DIR, name)
    os.makedirs(path, exist_ok=True)
    for i in indices:
        fname, pts, lbl, city, origin = blocks[i]
        np.savez_compressed(os.path.join(path, fname), points=pts, labels=lbl, city=city, origin=origin)

save("train", tr); save("val", va); save("test", te)
print(f"Train: {len(tr)}, Val: {len(va)}, Test: {len(te)}")
