import os, glob
import numpy as np
from tqdm import tqdm

def get_label(filename):
    try: return int(os.path.basename(filename).split("_class")[-1].split(".npy")[0])
    except: return None

def load_points(file):
    try:
        pts = np.load(file, allow_pickle=True)
        return pts[:, :3] if pts.ndim == 2 and pts.shape[1] >= 3 else None
    except: return None

def rebuild(city, in_dir="objects_all", out_dir="reconstructed_scenes"):
    os.makedirs(out_dir, exist_ok=True)
    pts_list, lbl_list = [], []
    for f in tqdm(glob.glob(f"{in_dir}/{city}_*.npy"), desc=f"{city}"):
        pts, lbl = load_points(f), get_label(f)
        if pts is not None and lbl is not None and len(pts):
            pts_list.append(pts)
            lbl_list.append(np.full(len(pts), lbl))
    if not pts_list: return
    np.savez(os.path.join(out_dir, f"{city}_scene.npz"),
             points=np.vstack(pts_list), labels=np.hstack(lbl_list))

for c in ["paris", "lille1", "lille2"]:
    rebuild(c)