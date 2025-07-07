import os, sys, torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import numpy as np, random, json
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from dataset_blocks import BlockDataset
import pointnet2_ops.pointnet2_utils as pointnet2_utils

# === Data Augmentation ===
class Compose:  # Compose multiple transforms
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, points, labels):
        for t in self.transforms: points, labels = t(points, labels)
        return points, labels

class RandomRotation:  # Rotate around Z
    def __call__(self, points, labels):
        theta = np.random.uniform(0, 2*np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0, 0, 1]], dtype=np.float32)
        return (points @ R.T).astype(np.float32), labels

class RandomNoise:
    def __init__(self, std=0.01): self.std = std
    def __call__(self, pts, lbl): return (pts + np.random.normal(0, self.std, pts.shape)).astype(np.float32), lbl

class RandomScale:
    def __init__(self, rng=(0.95, 1.05)): self.rng = rng
    def __call__(self, pts, lbl): return (pts * np.random.uniform(*self.rng)).astype(np.float32), lbl

# === PointNet2 Architecture ===
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_c, mlp):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.convs = nn.ModuleList([nn.Conv2d(in_c if i==0 else mlp[i-1], m, 1) for i, m in enumerate(mlp)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(m) for m in mlp])
    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = pointnet2_utils.gather_operation(xyz.transpose(1,2), fps_idx).transpose(1,2)
        idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = pointnet2_utils.grouping_operation(xyz.transpose(1,2), idx) - new_xyz.transpose(1,2).unsqueeze(-1)
        grouped = grouped_xyz if points is None else torch.cat([grouped_xyz, pointnet2_utils.grouping_operation(points.transpose(1,2), idx)], dim=1)
        for conv, bn in zip(self.convs, self.bns): grouped = F.relu(bn(conv(grouped)))
        return new_xyz, torch.max(grouped, -1)[0].transpose(1, 2)

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_c, mlp):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_c if i==0 else mlp[i-1], m, 1) for i, m in enumerate(mlp)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(m) for m in mlp])
    def forward(self, xyz1, xyz2, pts1, pts2):
        if xyz2.shape[1] == 1:
            interpolated = pts2.repeat(1, xyz1.shape[1], 1)
        else:
            dists, idx = pointnet2_utils.three_nn(xyz1, xyz2)
            weight = (1.0 / (dists + 1e-8)); weight /= weight.sum(dim=2, keepdim=True)
            interpolated = pointnet2_utils.three_interpolate(pts2.transpose(1,2), idx, weight).transpose(1,2)
        new_pts = torch.cat([pts1, interpolated], dim=-1) if pts1 is not None else interpolated
        for conv, bn in zip(self.convs, self.bns): new_pts = F.relu(bn(conv(new_pts.transpose(1,2)))).transpose(1,2)
        return new_pts

class PointNet2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3, [32, 32, 64])
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 67, [64, 64, 128])
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 131, [128, 128, 256])
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 259, [256, 256, 512])
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.cls = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5), nn.Conv1d(128, num_classes, 1)
        )
    def forward(self, xyz):
        l0 = xyz; p0 = None
        l1, p1 = self.sa1(l0, p0)
        l2, p2 = self.sa2(l1, p1)
        l3, p3 = self.sa3(l2, p2)
        l4, p4 = self.sa4(l3, p3)
        p3 = self.fp4(l3, l4, p3, p4)
        p2 = self.fp3(l2, l3, p2, p3)
        p1 = self.fp2(l1, l2, p1, p2)
        p0 = self.fp1(l0, l1, p0, p1)
        return F.log_softmax(self.cls(p0.transpose(1,2)), dim=1).transpose(1,2)

# === Training loop (simplified) ===
def train(model, loader, optimizer, criterion, device):
    model.train()
    for pts, lbl in tqdm(loader, desc='Training'):
        pts, lbl = pts.to(device).float(), lbl.to(device).long()
        optimizer.zero_grad()
        out = model(pts).view(-1, out.shape[-1])
        loss = criterion(out, lbl.view(-1))
        loss.backward(); optimizer.step()

# === Main (simplified) ===
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = BlockDataset('final_dataset/train', num_points=1024)
    train_ds.transform = Compose([RandomRotation(), RandomNoise(0.01), RandomScale()])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    model = PointNet2(num_classes=3).to(device)  # adapt num_classes
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.NLLLoss()
    for epoch in range(50):
        print(f"Epoch {epoch+1}"); train(model, train_loader, optimizer, criterion, device)

if __name__ == "__main__": main()
