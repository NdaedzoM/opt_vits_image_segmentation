# colab_training

Two things for running long ablation training on Colab Free without losing
work to disconnects, and for planning the run schedule with real numbers
instead of guesses.

## 1. Benchmark before you commit GPU hours

```python
from colab_training.benchmark import benchmark_epoch_time, project_run_time

sec_per_epoch = benchmark_epoch_time(model, train_loader, loss_fn, optimizer, device)

# e.g. 6 ablation configs, 1 seed each, 100 epochs
project_run_time(sec_per_epoch, num_epochs=100, num_configs=6, num_seeds=1)
```

Run this on the real model, real dataloader, real batch size/resolution,
on whatever GPU Colab hands you that session (T4 availability varies).
This is the number the whole timeline depends on — get it on day one for
both BraTS2020 and ADE20K, since they'll have very different sec/epoch.

## 2. Checkpoint to Drive so disconnects don't cost a run

```python
from google.colab import drive
drive.mount("/content/drive")

from colab_training.checkpoint import CheckpointManager

ckpt = CheckpointManager("/content/drive/MyDrive/uovit_e/<config_name>_<dataset>_seed<k>")
start_epoch, start_step = ckpt.resume_or_start(model, optimizer, scheduler)

for epoch in range(start_epoch, num_epochs):
    for step, batch in enumerate(train_loader):
        ...  # your existing training step
        if ckpt.should_autosave():
            ckpt.save_latest(model, optimizer, scheduler, epoch, step)

    val_metric = evaluate(...)
    ckpt.save_latest(model, optimizer, scheduler, epoch + 1, 0)
    if val_metric > best_so_far:
        best_so_far = val_metric
        ckpt.save_best(model, optimizer, scheduler, epoch + 1, 0, val_metric)
```

Re-running this same cell after a disconnect (fresh runtime, re-mount
Drive, re-run the setup cells) picks up exactly where it left off —
`resume_or_start` restores model/optimizer/scheduler weights and RNG state.

Give every (config, dataset, seed) combination its own `run_dir` — don't
reuse one directory across configs, or a resume will silently load the
wrong run's weights.

### Notes specific to Colab Free

- Idle disconnects happen even mid-session if the tab loses focus for too
  long; keep the tab active or use a keep-alive if your policy allows it.
- `autosave_minutes=10` (the default) bounds your worst-case loss to ~10
  minutes of compute. Lower it for expensive configs, but Drive write
  latency means going much below a minute or two isn't worth it.
- Drive has a file-count/quota ceiling on free accounts; this manager only
  ever keeps 3 files per run (`latest`, `latest_prev`, `best`), so it won't
  blow your quota even across ~30 runs.
