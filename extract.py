import os, numpy as np
from plyfile import PlyData
import pandas as pd, csv

BASE_DIR = "/home/yani/Projets/IA/stage_ub/data/training/taining_50_classes"
PLY_FILES = [
    {"ply": f"{BASE_DIR}/Paris.ply", "annotations": f"{BASE_DIR}/Paris_annotations.txt", "city": "paris"},
    {"ply": f"{BASE_DIR}/Lille1.ply", "annotations": f"{BASE_DIR}/Lille1_annotations.txt", "city": "lille1"},
    {"ply": f"{BASE_DIR}/Lille2.ply", "annotations": f"{BASE_DIR}/Lille2_annotations.txt", "city": "lille2"}
]

OUTPUT_DIR = "objects_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_annotations(path):
    with open(path, 'r') as f:
        return pd.DataFrame([[r[0], r[1], r[2], ", ".join(r[3:]) if len(r)>3 else ""]
                              for r in csv.reader(f) if len(r) >= 3],
                             columns=["label", "class_id", "class_name", "comment"])

def extract_objects(ply_path, ann_path, city):
    try:
        ann = read_annotations(ann_path)
        ply = PlyData.read(ply_path)['vertex'].data
        pts = np.stack([ply['x'], ply['y'], ply['z']], axis=-1)
        lbl = np.array(ply['label'])
    except Exception as e:
        print(f"Erreur chargement {city}: {e}"); return 0

    count = 0
    for _, row in ann.iterrows():
        try:
            mask = lbl == int(row['label'])
            obj_pts = pts[mask]
            if len(obj_pts) < 10: continue
            name = row['class_name'].replace("/", "_").replace(" ", "_").replace(".", "")
            fn = f"{city}_{row['label']}_{name}_class{row['class_id']}.npy"
            np.save(os.path.join(OUTPUT_DIR, fn), obj_pts)
            count += 1
        except: continue
    return count

print("🚀 Extraction d'objets depuis les PLY...")
total = 0
for f in PLY_FILES:
    c = extract_objects(f['ply'], f['annotations'], f['city'])
    print(f"{f['city']}: {c} objets")
    total += c
print(f"Total extrait: {total}")
