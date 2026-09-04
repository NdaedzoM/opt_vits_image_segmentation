"""Dataset classes reading the output of data/brats2020_preprocess.py."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedSliceDataset(Dataset):
    """Reads train/*.npz slices: image (C,H,W) float32, label (H,W) int64."""

    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz slices found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        return torch.from_numpy(d["image"]), torch.from_numpy(d["label"]).long()


class PreprocessedVolumeDataset(Dataset):
    """Reads val/ or test/ *.npz volumes: image (C,D,H,W) float32,
    label (D,H,W) int64. One item per patient -- used for volume-level
    evaluation (mean Dice per volume, then averaged across patients).
    """

    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz volumes found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        patient_id = self.files[idx].stem
        return patient_id, torch.from_numpy(d["image"]), torch.from_numpy(d["label"]).long()


class SyntheticSegDataset(Dataset):
    """Random tensors shaped like real input -- isolates model speed from
    data-loading speed, useful before a real preprocessed dataset exists.
    """

    def __init__(self, img_size, in_chans, num_classes, length=200):
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.randn(self.in_chans, self.img_size, self.img_size)
        label = torch.randint(0, self.num_classes, (self.img_size, self.img_size))
        return image, label
