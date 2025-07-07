import os, glob
import open3d as o3d
import numpy as np

def normalize_ply(input_path, output_path, scale=1.0):
    try:
        pcd = o3d.io.read_point_cloud(input_path)
        if len(pcd.points) == 0: return False
        pts = np.asarray(pcd.points) - np.mean(np.asarray(pcd.points), axis=0)
        if scale != 1.0: pts *= scale
        pcd.points = o3d.utility.Vector3dVector(pts)
        if len(pcd.colors): pcd.colors = pcd.colors
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return o3d.io.write_point_cloud(output_path, pcd)
    except: return False

def main():
    input_dir = "visualizations_real"
    output_dir = "visualizations_real_normalized"
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(f"{input_dir}/*.ply") + [f for f in [
        "data/test/lhd_infer.ply", "ply_exports/lhd_infer_segmented.ply", "lhd.ply",
        "data/lhd.ply", "data/test/lhd.ply", "ply_exports/lhd.ply"] if os.path.exists(f)]

    for f in files:
        name = os.path.basename(f)
        out = os.path.join(output_dir, f"normalized_{'lhd_original' if 'lhd' in name.lower() else name}")
        normalize_ply(f, out)

if __name__ == "__main__": main()
