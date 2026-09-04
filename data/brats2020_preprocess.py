"""Preprocess BraTS2020 (MICCAI_BraTS2020_TrainingData) into 2D axial
slices suitable for the SwinUnet / UOViT-E training pipeline.

Expected input layout (the standard BraTS2020 / Kaggle download):

    <data-root>/
        BraTS20_Training_001/
            BraTS20_Training_001_flair.nii[.gz]
            BraTS20_Training_001_t1.nii[.gz]
            BraTS20_Training_001_t1ce.nii[.gz]
            BraTS20_Training_001_t2.nii[.gz]
            BraTS20_Training_001_seg.nii[.gz]
        BraTS20_Training_002/
            ...

Label remap (BraTS's raw labels are {0, 1, 2, 4} -- 3 is unused): this
script remaps to a contiguous single-label multiclass problem
{0: background, 1: NCR/NET, 2: edema, 3: enhancing tumor}, matching the
single-label CrossEntropy + Dice setup the proposal specifies and the
SwinUnet baseline's `CrossEntropyLoss` import. This is NOT the same as
BraTS's official overlapping evaluation regions (whole tumor / tumor
core / enhancing tumor); if the report needs those, derive them from
this 4-class mask afterward (WT = classes 1+2+3, TC = classes 1+3,
ET = class 3) rather than re-preprocessing.

Output layout:

    <output-dir>/
        train/<patient_id>_slice<NNN>.npz   -- one 2D slice each:
            image: (C, img_size, img_size) float32
            label: (img_size, img_size) int64
        val/<patient_id>.npz                -- one full volume each:
            image: (C, D, img_size, img_size) float32
            label: (D, img_size, img_size) int64
        test/<patient_id>.npz                -- same layout as val/
        splits.json                          -- patient IDs per split
        preprocessing_config.json            -- exact settings used

Volumes are kept whole for val/test (rather than split into per-slice
files like train/) so volume-level metrics -- HD95 in particular, which
is a boundary distance metric and is only meaningful reassembled across
slices -- can be computed correctly; slicing them up only for training
avoids one giant volume dominating a training batch's gradient.

Patient-level split: slices from one patient never cross train/val/test,
otherwise near-duplicate adjacent slices leak between splits and inflate
reported performance.
"""

import argparse
import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.transform import resize

MODALITY_SUFFIXES = {"flair": "_flair", "t1": "_t1", "t1ce": "_t1ce", "t2": "_t2"}
SEG_SUFFIX = "_seg"

# BraTS raw label -> contiguous single-label class index
LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}


def find_patient_dirs(data_root):
    data_root = Path(data_root)
    dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("BraTS20"))
    if not dirs:
        raise FileNotFoundError(
            f"no BraTS20_Training_* directories found under {data_root} -- "
            "check --data-root points at MICCAI_BraTS2020_TrainingData"
        )
    return dirs


def _find_modality_file(patient_dir, suffix):
    for ext in (".nii.gz", ".nii"):
        candidate = patient_dir / f"{patient_dir.name}{suffix}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no {suffix} file found in {patient_dir}")


def load_patient_volumes(patient_dir, modalities):
    images = []
    for m in modalities:
        path = _find_modality_file(patient_dir, MODALITY_SUFFIXES[m])
        images.append(nib.load(str(path)).get_fdata(dtype=np.float32))
    seg_path = _find_modality_file(patient_dir, SEG_SUFFIX)
    seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)
    return np.stack(images, axis=0), seg  # (C, H, W, D), (H, W, D)


def remap_labels(seg):
    out = np.zeros_like(seg, dtype=np.int64)
    for raw, new in LABEL_MAP.items():
        out[seg == raw] = new
    return out


def normalize_modality(vol):
    """Foreground-only z-score normalisation, standard for BraTS: the
    background is a large constant-zero region and including it in the
    mean/std would wash out the actual tissue intensity distribution.
    """
    foreground = vol[vol > 0]
    if foreground.size == 0:
        return vol
    lo, hi = np.percentile(foreground, [1, 99])
    vol = np.clip(vol, lo, hi)
    mean, std = foreground.mean(), foreground.std()
    std = std if std > 1e-6 else 1.0
    out = (vol - mean) / std
    out[vol == 0] = 0.0
    return out.astype(np.float32)


def resize_slice(img_chw, label_hw, img_size):
    """img_chw: (C, H, W) float32. label_hw: (H, W) int64.
    Bilinear (anti-aliased) for the image, nearest-neighbour for the
    label -- interpolating a label mask with anything but nearest-neighbour
    invents fractional/blended class values that don't exist.
    """
    c = img_chw.shape[0]
    resized_img = np.stack([
        resize(img_chw[i], (img_size, img_size), order=1, mode="constant",
               anti_aliasing=True, preserve_range=True)
        for i in range(c)
    ]).astype(np.float32)
    resized_label = resize(label_hw, (img_size, img_size), order=0, mode="constant",
                            anti_aliasing=False, preserve_range=True).astype(np.int64)
    return resized_img, resized_label


def slice_has_brain(img_chw, min_nonzero_frac=0.01):
    nonzero_frac = (img_chw[0] != 0).mean()
    return nonzero_frac >= min_nonzero_frac


def process_patient_to_slices(images, seg, img_size, min_nonzero_frac):
    """images: (C, H, W, D) raw. seg: (H, W, D) raw labels.
    Returns list of (image (C,img_size,img_size), label (img_size,img_size))
    for slices along the axial (D) axis that contain brain tissue.
    """
    images = np.stack([normalize_modality(images[c]) for c in range(images.shape[0])], axis=0)
    labels = remap_labels(seg)

    depth = images.shape[-1]
    slices = []
    for d in range(depth):
        img_slice = images[:, :, :, d]  # (C, H, W)
        label_slice = labels[:, :, d]   # (H, W)
        if not slice_has_brain(img_slice, min_nonzero_frac):
            continue
        img_r, label_r = resize_slice(img_slice, label_slice, img_size)
        slices.append((img_r, label_r))
    return slices


def process_patient_to_volume(images, seg, img_size):
    images = np.stack([normalize_modality(images[c]) for c in range(images.shape[0])], axis=0)
    labels = remap_labels(seg)
    depth = images.shape[-1]

    img_out, label_out = [], []
    for d in range(depth):
        img_r, label_r = resize_slice(images[:, :, :, d], labels[:, :, d], img_size)
        img_out.append(img_r)
        label_out.append(label_r)
    # (C, D, img_size, img_size), (D, img_size, img_size)
    return np.stack(img_out, axis=1), np.stack(label_out, axis=0)


def split_patients(patient_dirs, val_frac, test_frac, seed):
    ids = [p.name for p in patient_dirs]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * test_frac)))
    val_ids = set(ids[:n_val])
    test_ids = set(ids[n_val:n_val + n_test])
    train_ids = set(ids[n_val + n_test:])
    return train_ids, val_ids, test_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="path to MICCAI_BraTS2020_TrainingData")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--modalities", nargs="+", default=["flair"],
                         choices=list(MODALITY_SUFFIXES.keys()),
                         help="default: flair alone (1 channel), compatible with SwinUnet's "
                              "single-channel-to-3-channel repeat and ImageNet-pretrained weights. "
                              "Passing more than one stacks channels (e.g. flair t1ce t2) but then "
                              "config.MODEL.SWIN.IN_CHANS must match and ImageNet-pretrained weights "
                              "can no longer be loaded directly for the patch embedding layer.")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-nonzero-frac", type=float, default=0.01,
                         help="skip slices where less than this fraction of pixels is brain tissue "
                              "(drops near-empty top/bottom-of-skull slices)")
    parser.add_argument("--limit-patients", type=int, default=None,
                         help="process only the first N patients -- use for a quick smoke test")
    args = parser.parse_args()

    patient_dirs = find_patient_dirs(args.data_root)
    if args.limit_patients:
        patient_dirs = patient_dirs[:args.limit_patients]
    print(f"found {len(patient_dirs)} patients")

    train_ids, val_ids, test_ids = split_patients(patient_dirs, args.val_frac, args.test_frac, args.seed)
    print(f"split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test patients")

    output_dir = Path(args.output_dir)
    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    total_train_slices = 0
    for i, patient_dir in enumerate(patient_dirs):
        pid = patient_dir.name
        split = "train" if pid in train_ids else ("val" if pid in val_ids else "test")

        images, seg = load_patient_volumes(patient_dir, args.modalities)

        if split == "train":
            slices = process_patient_to_slices(images, seg, args.img_size, args.min_nonzero_frac)
            for j, (img, label) in enumerate(slices):
                np.savez_compressed(output_dir / "train" / f"{pid}_slice{j:03d}.npz",
                                     image=img, label=label)
            total_train_slices += len(slices)
            print(f"[{i + 1}/{len(patient_dirs)}] {pid} (train): {len(slices)} slices kept")
        else:
            vol_img, vol_label = process_patient_to_volume(images, seg, args.img_size)
            np.savez_compressed(output_dir / split / f"{pid}.npz", image=vol_img, label=vol_label)
            print(f"[{i + 1}/{len(patient_dirs)}] {pid} ({split}): volume {vol_img.shape} saved")

    with open(output_dir / "splits.json", "w") as f:
        json.dump({"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}, f, indent=2)

    with open(output_dir / "preprocessing_config.json", "w") as f:
        json.dump({
            "img_size": args.img_size,
            "modalities": args.modalities,
            "label_map": LABEL_MAP,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "min_nonzero_frac": args.min_nonzero_frac,
            "total_train_slices": total_train_slices,
        }, f, indent=2)

    print(f"done. {total_train_slices} total train slices written to {output_dir}")


if __name__ == "__main__":
    main()
