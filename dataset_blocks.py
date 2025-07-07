# dataset_blocks.py
import os
import numpy as np
from torch.utils.data import Dataset

def sample_points(points, labels, num_points=1024):
    """Échantillonne les points et leurs labels sans altération des étiquettes."""
    if len(points) >= num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
    else:
        idx = np.random.choice(len(points), num_points, replace=True)
    return points[idx], labels[idx]

class BlockDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_points=1024):
        self.root_dir = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith(".npz")])
        self.transform = transform
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = os.path.join(self.root_dir, self.files[idx])
        data = np.load(fpath)
        
        # Chargement avec vérification explicite
        points = data['points'].astype(np.float32)
        labels = data['labels'].astype(np.int64)
        
        # Validation des données
        assert len(points) == len(labels), f"Points/labels mismatch in {self.files[idx]}"
        
        points, labels = sample_points(points, labels, self.num_points)
        
        if self.transform:
            points, labels = self.transform(points, labels)
            
        return points, labels