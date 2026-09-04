"""Measure real epoch time on your actual model/dataloader/GPU before
committing Colab hours to a full run schedule.

Rule-of-thumb estimates for ADE20K/BraTS2020 training time are not reliable
enough to plan a month around. Run this against your real model, real
dataloader, and the real batch size/resolution you intend to train with,
on the actual GPU Colab assigns you, and use the printed numbers to build
the run schedule.

Usage (inside Colab, after building model/dataloader/optimizer):

    from colab_training.benchmark import benchmark_epoch_time, project_run_time

    sec_per_epoch = benchmark_epoch_time(model, dataloader, loss_fn, optimizer, device)
    project_run_time(sec_per_epoch, num_epochs=100, num_configs=6, num_seeds=1)
"""

import time

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _step(model, batch, loss_fn, optimizer, device):
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss


def benchmark_epoch_time(model, dataloader, loss_fn, optimizer, device,
                          warmup_iters=5, measure_iters=25):
    """Times `measure_iters` real training steps (after `warmup_iters` to let
    cudnn autotune settle) and extrapolates to seconds/epoch.
    """
    model.train()
    it = iter(dataloader)

    for _ in range(warmup_iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)
        _step(model, batch, loss_fn, optimizer, device)
    _sync()

    start = time.time()
    for _ in range(measure_iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)
        _step(model, batch, loss_fn, optimizer, device)
    _sync()
    elapsed = time.time() - start

    sec_per_iter = elapsed / measure_iters
    iters_per_epoch = len(dataloader)
    sec_per_epoch = sec_per_iter * iters_per_epoch

    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"peak GPU memory: {peak_mem_gb:.2f} GB")

    print(f"sec/iter:    {sec_per_iter:.3f}")
    print(f"iters/epoch: {iters_per_epoch}")
    print(f"sec/epoch:   {sec_per_epoch:.1f}  ({sec_per_epoch / 60:.1f} min)")
    return sec_per_epoch


def project_run_time(sec_per_epoch, num_epochs, num_configs, num_seeds=1,
                      colab_session_hours=11.0):
    """Extrapolates a single benchmarked epoch time to the full ablation
    schedule and reports how many Colab sessions that requires.
    """
    total_hours = (sec_per_epoch * num_epochs * num_configs * num_seeds) / 3600
    sessions_needed = total_hours / colab_session_hours

    print(f"Total training time: {total_hours:.1f} GPU-hours")
    print(f"  = {num_configs} config(s) x {num_seeds} seed(s) x {num_epochs} epochs")
    print(f"Colab sessions needed (~{colab_session_hours}h usable each): {sessions_needed:.1f}")
    return total_hours
