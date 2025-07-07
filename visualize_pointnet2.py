import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt, seaborn as sns, os
from sklearn.metrics import confusion_matrix, classification_report
import open3d as o3d
from dataset_blocks import BlockDataset
import pointnet2_ops.pointnet2_utils as pointnet2_utils

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.group_all = npoint, radius, nsample, group_all
        self.mlp_convs = nn.ModuleList([nn.Conv2d(in_channel if i == 0 else mlp[i - 1], out, 1) for i, out in enumerate(mlp)])
        self.mlp_bns = nn.ModuleList([nn.BatchNorm2d(out) for out in mlp])

    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        if self.group_all:
            new_xyz = xyz[:, 0:1, :]
            new_points = torch.cat([xyz, points], dim=2) if points is not None else xyz
            new_points = new_points.permute(0, 2, 1).unsqueeze(2)
        else:
            xyz_flipped = xyz.transpose(1, 2).contiguous()
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, fps_idx).transpose(1, 2).contiguous()
            idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = pointnet2_utils.grouping_operation(xyz_flipped, idx)
            grouped_xyz_norm = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
            if points is not None:
                points_flipped = points.transpose(1, 2).contiguous()
                grouped_points = pointnet2_utils.grouping_operation(points_flipped, idx)
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=1)
            else:
                new_points = grouped_xyz_norm
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, -1)[0].transpose(1, 2)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList([nn.Conv1d(in_channel if i == 0 else mlp[i - 1], out, 1) for i, out in enumerate(mlp)])
        self.mlp_bns = nn.ModuleList([nn.BatchNorm1d(out) for out in mlp])

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, _ = xyz1.shape
        _, M, _ = xyz2.shape
        if M == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dist, idx = pointnet2_utils.three_nn(xyz1, xyz2)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = pointnet2_utils.three_interpolate(points2.transpose(1, 2), idx, weight).transpose(1, 2)
        new_points = torch.cat([points1, interpolated_points], dim=-1) if points1 is not None else interpolated_points
        new_points = new_points.transpose(1, 2)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points.transpose(1, 2)

class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 3, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(1280, [256, 256])
        self.fp2 = PointNetFeaturePropagation(384, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        l0_xyz, l0_points = xyz, None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        x = F.relu(self.bn1(self.conv1(l0_points.transpose(1, 2))))
        x = self.drop1(x)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1).transpose(1, 2)

if __name__ == "__main__":
    model = PointNet2SemSeg(num_classes=6).eval()
    dummy_input = torch.randn(1, 1024, 3)
    output = model(dummy_input)
    print("Output shape:", output.shape)
