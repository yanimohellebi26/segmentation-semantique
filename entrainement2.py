import os, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_blocks import BlockDataset
import pointnet2_ops.pointnet2_utils as pointnet2_utils

# === Transformations ===
class Compose:  # Compose multiple transforms
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, pts, lbl):
        for t in self.transforms: pts, lbl = t(pts, lbl)
        return pts, lbl

class RandomRotation:  # Rotation Z
    def __call__(self, pts, lbl):
        th = np.random.uniform(0, 2*np.pi)
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        return (pts @ R.T).astype(np.float32), lbl

class RandomNoise:  # Bruit gaussien
    def __init__(self, std=0.01): self.std = std
    def __call__(self, pts, lbl): return (pts + np.random.normal(0, self.std, pts.shape)).astype(np.float32), lbl

class RandomScale:
    def __init__(self, rng=(0.95, 1.05)): self.rng = rng
    def __call__(self, pts, lbl): return (pts * np.random.uniform(*self.rng)).astype(np.float32), lbl

# === Dataset équilibré ===
class BalancedDataset:
    def __init__(self, base_ds, label_map, sampler):
        self.ds, self.map, self.sampler = base_ds, label_map, sampler
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        pts, lbl = self.ds[i]
        lbl_mapped = np.vectorize(self.map.get)(lbl)
        if self.sampler: pts, lbl_mapped = self.sampler(pts, lbl_mapped)
        return pts, lbl_mapped

# === Architecture PointNet2 réduite ===
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_c, mlp):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.convs = nn.ModuleList([nn.Conv2d(in_c if i==0 else mlp[i-1], m, 1) for i, m in enumerate(mlp)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(m) for m in mlp])
    def forward(self, xyz, points):
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = pointnet2_utils.gather_operation(xyz.transpose(1,2), fps_idx).transpose(1,2)
        idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped = pointnet2_utils.grouping_operation(xyz.transpose(1,2), idx) - new_xyz.transpose(1,2).unsqueeze(-1)
        if points is not None:
            grouped_pts = pointnet2_utils.grouping_operation(points.transpose(1,2), idx)
            grouped = torch.cat([grouped, grouped_pts], dim=1)
        for conv, bn in zip(self.convs, self.bns): grouped = F.relu(bn(conv(grouped)))
        return new_xyz, torch.max(grouped, -1)[0].transpose(1, 2)

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_c, mlp):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_c if i==0 else mlp[i-1], m, 1) for i, m in enumerate(mlp)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(m) for m in mlp])
    def forward(self, xyz1, xyz2, pts1, pts2):
        d, i = pointnet2_utils.three_nn(xyz1, xyz2)
        w = (1.0 / (d + 1e-8)); w /= w.sum(2, keepdim=True)
        interp = pointnet2_utils.three_interpolate(pts2.transpose(1,2), i, w).transpose(1,2)
        new_pts = torch.cat([pts1, interp], dim=-1) if pts1 is not None else interp
        for conv, bn in zip(self.convs, self.bns): new_pts = F.relu(bn(conv(new_pts.transpose(1,2)))).transpose(1,2)
        return new_pts

class PointNet2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 131, [128, 128, 256])
        self.fp1 = PointNetFeaturePropagation(384, [256, 128])
        self.fp0 = PointNetFeaturePropagation(131, [128, 128])
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5), nn.Conv1d(128, num_classes, 1)
        )
    def forward(self, xyz):
        l0 = xyz; p0 = None
        l1, p1 = self.sa1(l0, p0)
        l2, p2 = self.sa2(l1, p1)
        p1 = self.fp1(l1, l2, p1, p2)
        p0 = self.fp0(l0, l1, p0, p1)
        return F.log_softmax(self.classifier(p0.transpose(1,2)), dim=1).transpose(1,2)

# === Entraînement simplifié ===
def train(model, loader, opt, loss_fn, device):
    model.train()
    for pts, lbl in tqdm(loader, desc='Training'):
        pts, lbl = pts.to(device).float(), lbl.to(device).long()
        opt.zero_grad()
        pred = model(pts).view(-1, pred.shape[-1])
        loss = loss_fn(pred, lbl.view(-1))
        loss.backward(); opt.step()

# === Main simplifié ===
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_ds = BlockDataset('final_dataset/train', num_points=1024)
    base_ds.transform = Compose([RandomRotation(), RandomNoise(), RandomScale()])
    dummy_map = {i: i for i in range(3)}
    balanced_ds = BalancedDataset(base_ds, dummy_map, None)
    loader = DataLoader(balanced_ds, batch_size=8, shuffle=True, num_workers=4)
    model = PointNet2(num_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()
    for ep in range(50):
        print(f"Epoch {ep+1}"); train(model, loader, opt, loss_fn, device)

if __name__ == "__main__": main()
