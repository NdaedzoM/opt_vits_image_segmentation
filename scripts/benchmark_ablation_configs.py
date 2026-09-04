"""Benchmark every UOViT-E ablation config: params, GFLOPs (single forward,
batch=1), and measured sec/epoch + projected GPU-hours for the full run
schedule.

This is the script referenced in colab_training/README.md -- run it in
Colab on whatever GPU that session gets, against either real preprocessed
data (--data-dir, pointing at a brats2020_preprocess.py train/ output) or
synthetic random tensors (default, no --data-dir) for a pure model-speed
comparison before preprocessing is done.

Usage in Colab:

    !git clone https://github.com/NdaedzoM/opt_vits_image_segmentation
    %cd opt_vits_image_segmentation
    !python scripts/benchmark_ablation_configs.py --img-size 224 --num-classes 4 \
        --in-chans 1 --batch-size 8 --num-epochs 100
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from networks.vision_transformer_uovit import build_uovit_e, ABLATION_CONFIGS
from colab_training.benchmark import benchmark_epoch_time, project_run_time
from data.datasets import SyntheticSegDataset, PreprocessedSliceDataset


def measure_gflops(model, img_size, in_chans, device):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    x = torch.randn(1, in_chans, img_size, img_size, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    model.train()
    return fc.get_total_flops() / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--in-chans", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--depths", type=int, nargs=4, default=[2, 2, 2, 2])
    parser.add_argument("--num-heads", type=int, nargs=4, default=[3, 6, 12, 24])
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-dir", default=None,
                         help="path to a brats2020_preprocess.py train/ directory of .npz slices; "
                              "omit to benchmark on synthetic random data instead")
    parser.add_argument("--num-epochs", type=int, default=100,
                         help="epochs per config, used only to project total GPU-hours")
    parser.add_argument("--num-seeds", type=int, default=1,
                         help="seeds per config, used only to project total GPU-hours")
    parser.add_argument("--configs", nargs="+", default=list(ABLATION_CONFIGS.keys()),
                         choices=list(ABLATION_CONFIGS.keys()))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU visible -- sec/epoch numbers below are CPU-speed and will NOT "
              "reflect real Colab training time. GFLOPs and param counts are still valid.")

    if args.data_dir:
        dataset = PreprocessedSliceDataset(args.data_dir)
        print(f"dataset: {len(dataset)} real slices from {args.data_dir}")
    else:
        dataset = SyntheticSegDataset(args.img_size, args.in_chans, args.num_classes)
        print("dataset: synthetic random tensors (pass --data-dir for a real-data benchmark)")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loss_fn = nn.CrossEntropyLoss()

    results = {}
    for name in args.configs:
        print(f"\n=== {name} ===")
        model = build_uovit_e(
            img_size=args.img_size, in_chans=args.in_chans, num_classes=args.num_classes,
            embed_dim=args.embed_dim, depths=tuple(args.depths), num_heads=tuple(args.num_heads),
            window_size=args.window_size, config_name=name,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        gflops = measure_gflops(model, args.img_size, args.in_chans, device)
        print(f"params: {n_params:,}  GFLOPs (batch=1): {gflops:.2f}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        sec_per_epoch = benchmark_epoch_time(model, dataloader, loss_fn, optimizer, device)
        total_hours = project_run_time(sec_per_epoch, args.num_epochs, num_configs=1, num_seeds=args.num_seeds)

        results[name] = dict(params=n_params, gflops=gflops, sec_per_epoch=sec_per_epoch,
                              gpu_hours_this_config=total_hours)

    print("\n=== summary ===")
    header = f"{'config':<14}{'params':>12}{'GFLOPs':>10}{'sec/epoch':>12}{'GPU-h (this cfg)':>18}"
    print(header)
    total_hours = 0.0
    for name, r in results.items():
        print(f"{name:<14}{r['params']:>12,}{r['gflops']:>10.2f}{r['sec_per_epoch']:>12.1f}"
              f"{r['gpu_hours_this_config']:>18.2f}")
        total_hours += r["gpu_hours_this_config"]
    print(f"\ntotal projected GPU-hours across the {len(results)} config(s) benchmarked, "
          f"{args.num_epochs} epochs x {args.num_seeds} seed(s) each: {total_hours:.1f}")


if __name__ == "__main__":
    main()
