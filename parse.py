import os, glob, shutil
import numpy as np
from collections import defaultdict

INPUT_DIR, OUTPUT_DIR = "objects", "objects_10classes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_MAPPING = {
    "vegetation": [304020000, 304000000],
    "building": [202000000, 202010000, 202020000],
    "street_furniture": [303050500, 303050000, 303050100],
    "signage": [302020700, 302020900, 302020000],
    "infrastructure": [302030600, 302030200, 302040700, 302020600],
    "parked_vehicles": [303030204, 303030200, 303030201],
    "pedestrian": [303020000, 303020100],
    "transport": [301020100, 301010000],
    "terrain": [306000000, 203000000],
    "other": [100000000]
}

CLASS_TO_SUPER = {v: k for k, lst in CLASS_MAPPING.items() for v in lst}
SUPER_TO_ID = {k: i for i, k in enumerate(CLASS_MAPPING)}

files = glob.glob(f"{INPUT_DIR}/*.npy")
counts = defaultdict(int)

for f in files:
    try:
        id_str = f.split("_class")[-1].replace(".npy", "")
        class_id = int(id_str)
        super_cls = CLASS_TO_SUPER.get(class_id, "other")
        new_id = SUPER_TO_ID[super_cls]
        new_name = f.split("_class")[0] + f"_class{new_id}.npy"
        shutil.copy2(f, os.path.join(OUTPUT_DIR, new_name))
        counts[new_id] += 1
    except: continue

print("Conversion terminée.")
for k in sorted(counts):
    super_cls = [s for s, i in SUPER_TO_ID.items() if i == k][0]
    print(f"Classe {k} ({super_cls}): {counts[k]} fichiers")