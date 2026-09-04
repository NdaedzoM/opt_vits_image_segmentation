"""Drive-backed checkpointing for long Colab training runs.

Colab Free introduces two failure modes that will otherwise cost you full
training runs:
  - the runtime disconnects mid-epoch (idle timeout, ~12h session cap, GPU
    revoked under demand)
  - you close the tab / lose network and need to resume from a fresh
    runtime later

Checkpoints are written to a directory on Google Drive (survives runtime
resets) on a wall-clock interval, not just at epoch boundaries, so a crash
never costs more than `autosave_minutes` of work. Writes are atomic
(write to a temp file, then os.replace) so a crash mid-save can't corrupt
the checkpoint you'd resume from, and the previous "latest" is kept as a
backup in case the newest write is bad.

Usage (inside your training loop):

    ckpt = CheckpointManager("/content/drive/MyDrive/uovit_e/run1")
    start_epoch, start_step = ckpt.resume_or_start(model, optimizer, scheduler)

    for epoch in range(start_epoch, num_epochs):
        for step, batch in enumerate(dataloader):
            ...
            if ckpt.should_autosave():
                ckpt.save_latest(model, optimizer, scheduler, epoch, step)

        val_metric = evaluate(...)
        ckpt.save_latest(model, optimizer, scheduler, epoch + 1, 0)
        if val_metric > best_so_far:
            ckpt.save_best(model, optimizer, scheduler, epoch + 1, 0, val_metric)
"""

import os
import random
import shutil
import time

import numpy as np
import torch


class CheckpointManager:
    def __init__(self, run_dir, autosave_minutes=10):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.autosave_minutes = autosave_minutes
        self._last_save = time.time()

    def _path(self, tag):
        return os.path.join(self.run_dir, f"ckpt_{tag}.pt")

    def save(self, tag, model, optimizer, scheduler, epoch, step, extra=None):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "step": step,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
            "extra": extra or {},
        }
        final_path = self._path(tag)
        tmp_path = final_path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, final_path)  # atomic on the same filesystem
        self._last_save = time.time()

    def save_latest(self, model, optimizer, scheduler, epoch, step, extra=None):
        latest = self._path("latest")
        if os.path.exists(latest):
            shutil.copy2(latest, self._path("latest_prev"))
        self.save("latest", model, optimizer, scheduler, epoch, step, extra)

    def save_best(self, model, optimizer, scheduler, epoch, step, metric, extra=None):
        extra = dict(extra or {})
        extra["metric"] = metric
        self.save("best", model, optimizer, scheduler, epoch, step, extra)

    def should_autosave(self):
        return (time.time() - self._last_save) >= self.autosave_minutes * 60

    def load(self, tag, model, optimizer=None, scheduler=None, map_location=None):
        path = self._path(tag)
        if not os.path.exists(path):
            return None
        # weights_only=False: checkpoints here always come from this same
        # CheckpointManager (never an untrusted third-party file) and
        # deliberately carry non-tensor state (RNG state, python ints)
        # that torch's default weights_only=True (PyTorch >= 2.6) refuses
        # to unpickle.
        state = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_torch"])
        if state.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        np.random.set_state(state["rng_numpy"])
        random.setstate(state["rng_python"])
        return state

    def get_best_metric(self, default=0.0):
        """Reads the metric stored by save_best() without touching model/
        optimizer state -- use on resume to seed a run's `best_so_far` so a
        later, worse epoch after a disconnect can't overwrite a genuinely
        better 'best' checkpoint from before the restart.
        """
        path = self._path("best")
        if not os.path.exists(path):
            return default
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            return state.get("extra", {}).get("metric", default)
        except Exception as e:
            print(f"[checkpoint] failed to read best metric: {e}")
            return default

    def resume_or_start(self, model, optimizer=None, scheduler=None, map_location=None):
        """Try 'latest', fall back to 'latest_prev' if the newest write is corrupt.

        Returns (start_epoch, start_step), (0, 0) if there is nothing to resume.
        """
        for tag in ("latest", "latest_prev"):
            try:
                state = self.load(tag, model, optimizer, scheduler, map_location)
                if state is not None:
                    print(f"[checkpoint] resumed from '{tag}' at epoch {state['epoch']}")
                    return state["epoch"], state["step"]
            except Exception as e:
                print(f"[checkpoint] failed to load '{tag}': {e}")
        print("[checkpoint] no valid checkpoint found, starting from scratch")
        return 0, 0
