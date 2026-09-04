"""Train one UOViT-E ablation config on preprocessed BraTS2020 data, with
Drive checkpointing so a Colab disconnect never costs more than
--autosave-minutes of work.

Usage in Colab:

    from google.colab import drive
    drive.mount("/content/drive")

    !git clone https://github.com/NdaedzoM/opt_vits_image_segmentation
    %cd opt_vits_image_segmentation

    !python scripts/train_uovit_e.py \
        --config-name uovit_e \
        --train-dir /path/to/brats_out/train \
        --val-dir   /path/to/brats_out/val \
        --img-size 224 --in-chans 1 --num-classes 4 \
        --run-dir /content/drive/MyDrive/uovit_e/uovit_e_brats_seed0

Re-running the same command after a disconnect (fresh runtime, re-mount
Drive, re-clone/re-run this cell) resumes from --run-dir automatically --
see colab_training/README.md for how CheckpointManager.resume_or_start works.

Give every (config, dataset, seed) combination its own --run-dir; reusing
one directory across configs will silently resume the wrong run's weights.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from networks.vision_transformer_uovit import build_uovit_e, ABLATION_CONFIGS
from data.datasets import PreprocessedSliceDataset, PreprocessedVolumeDataset
from data.losses import DiceCELoss, mean_dice_score
from colab_training.checkpoint import CheckpointManager


def evaluate(model, val_loader, num_classes, device, max_slice_batch=8):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for patient_id, image, label in val_loader:
            # image: (1, C, D, H, W), label: (1, D, H, W) -- one volume per item
            image, label = image[0].to(device), label[0].to(device)
            C, D, H, W = image.shape
            image = image.permute(1, 0, 2, 3)  # (D, C, H, W)

            preds_all = []
            for start in range(0, D, max_slice_batch):
                batch = image[start:start + max_slice_batch]
                logits = model(batch)
                preds_all.append(logits)
            logits_all = torch.cat(preds_all, dim=0)  # (D, num_classes, H, W)

            score = mean_dice_score(logits_all, label, num_classes)
            dice_scores.append(score)
    model.train()
    return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", required=True, choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--train-dir", required=True, help="brats2020_preprocess.py train/ output")
    parser.add_argument("--val-dir", required=True, help="brats2020_preprocess.py val/ output")
    parser.add_argument("--run-dir", required=True, help="Drive directory for this run's checkpoints")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--in-chans", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--depths", type=int, nargs=4, default=[2, 2, 2, 2])
    parser.add_argument("--num-heads", type=int, nargs=4, default=[3, 6, 12, 24])
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-dice", type=float, default=0.5)
    parser.add_argument("--lambda-ce", type=float, default=0.5)
    parser.add_argument("--autosave-minutes", type=float, default=10)
    parser.add_argument("--eval-every", type=int, default=5, help="run validation every N epochs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = PreprocessedSliceDataset(args.train_dir)
    val_ds = PreprocessedVolumeDataset(args.val_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f"train slices: {len(train_ds)}  val volumes: {len(val_ds)}")

    model = build_uovit_e(
        img_size=args.img_size, in_chans=args.in_chans, num_classes=args.num_classes,
        embed_dim=args.embed_dim, depths=tuple(args.depths), num_heads=tuple(args.num_heads),
        window_size=args.window_size, config_name=args.config_name,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"config: {args.config_name}  params: {n_params:,}")

    loss_fn = DiceCELoss(args.num_classes, args.lambda_dice, args.lambda_ce)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt = CheckpointManager(args.run_dir, autosave_minutes=args.autosave_minutes)
    start_epoch, _ = ckpt.resume_or_start(model, optimizer, scheduler, map_location=device)
    best_dice = ckpt.get_best_metric(default=0.0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for step, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if ckpt.should_autosave():
                ckpt.save_latest(model, optimizer, scheduler, epoch, step)

        scheduler.step()
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"epoch {epoch + 1}/{args.epochs}  loss {avg_loss:.4f}  ({time.time() - t0:.1f}s)")

        ckpt.save_latest(model, optimizer, scheduler, epoch + 1, 0)

        if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.epochs:
            val_dice = evaluate(model, val_loader, args.num_classes, device)
            print(f"  val mean Dice: {val_dice:.4f}")
            if val_dice > best_dice:
                best_dice = val_dice
                ckpt.save_best(model, optimizer, scheduler, epoch + 1, 0, val_dice)

    print(f"done. best val mean Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
