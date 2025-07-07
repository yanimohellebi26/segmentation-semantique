import os, sys, time, argparse
import torch, numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from model import PointNet2SemSeg

COLORS = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:, :3]

def load_model(model_path, device='cuda'):
    ckpt = torch.load(model_path, map_location=device)
    model = PointNet2SemSeg(num_classes=ckpt['num_classes']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['idx_to_label']

def preprocess_points(points):
    pts = points - points.mean(0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    return pts / scale if scale > 0 else pts

def segment_points(model, points, chunk_size=1024, device='cuda'):
    preds, all_idx = [], []
    for i in tqdm(range(0, len(points), chunk_size)):
        chunk = points[i:i+chunk_size]
        idx_range = np.arange(i, i+len(chunk))
        pad = chunk if len(chunk)==chunk_size else np.vstack([chunk, chunk[-1:].repeat(chunk_size - len(chunk), 0)])
        tensor = torch.tensor(pad).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)[0, :len(chunk)].argmax(1).cpu().numpy()
        preds.append(out); all_idx.append(idx_range)
    full_pred = np.zeros(len(points), dtype=int)
    full_pred[np.concatenate(all_idx)] = np.concatenate(preds)
    return full_pred

def knn_refinement(points, predictions, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(points, predictions)
    return knn.predict(points)

def save_colored_ply(points, labels, output_path):
    colors = np.array([COLORS[l % len(COLORS)] for l in labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)

def visualize(points, labels):
    colors = np.array([COLORS[l % len(COLORS)] for l in labels])
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    ax.set_axis_off(); plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ply_file')
    parser.add_argument('--model', default='best_model.pth')
    parser.add_argument('--output', default='results')
    parser.add_argument('--knn', action='store_true')
    args = parser.parse_args()

    raw_points = np.asarray(o3d.io.read_point_cloud(args.ply_file).points)
    points = preprocess_points(raw_points)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, label_map = load_model(args.model, device)

    start = time.time()
    labels = segment_points(model, points, 1024, device)
    if args.knn:
        labels = knn_refinement(points, labels)
    print(f"Done in {time.time()-start:.1f}s")

    stem = Path(args.ply_file).stem
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, f"{stem}_labels.npy"), labels)
    save_colored_ply(raw_points, labels, os.path.join(args.output, f"{stem}_labeled.ply"))
    visualize(points, labels)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv = [__file__, input("Chemin vers le fichier PLY: ")]
    main()
