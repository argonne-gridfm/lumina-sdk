import json
import os
from pathlib import Path
import socket
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ThroughputTracker:
    """Measures training throughput (samples/sec) over a configurable window.

    After a configurable warmup period, records per-step timing for a fixed
    number of measurement steps, then computes and logs the mean throughput
    across all DDP ranks.  Results are optionally logged to W&B and written
    as JSON metadata.

    Args:
        config (dict): Full training configuration. Throughput settings are
            read from ``config["training"]``: ``throughput_enabled``,
            ``throughput_warmup_steps``, ``throughput_measure_steps``.
        world_size (int): Total number of DDP processes.
        global_rank (int): Rank of the current process.
        get_global_step (callable): Zero-argument callable returning the
            current global step count (used as W&B x-axis).
        wandb_enabled (bool): Whether to log metrics to W&B.
    """

    def __init__(self, config, world_size, global_rank, get_global_step, wandb_enabled=False):
        self.config = config or {}
        self.world_size = world_size
        self.global_rank = global_rank
        self._get_global_step = get_global_step or (lambda: 0)
        self.wandb_enabled = bool(wandb_enabled) and WANDB_AVAILABLE

        training_config = self.config.get("training", {})
        self.enabled = training_config.get("throughput_enabled", False)
        self.warmup_steps = max(0, int(training_config.get("throughput_warmup_steps", 100)))
        self.measure_steps = max(0, int(training_config.get("throughput_measure_steps", 200)))
        if self.measure_steps == 0:
            self.enabled = False

        self.has_run = False
        self.step_index = 0
        self.measure_started = False
        self.measure_count = 0
        self.samples = []
        self.metadata_written = False
        self.loader_config = self.config.get("loader", {})

    def set_wandb_enabled(self, enabled):
        """Enable or disable W&B logging for throughput metrics.

        Args:
            enabled (bool): ``True`` to enable W&B logging (only effective
                if ``wandb`` is installed).
        """
        self.wandb_enabled = bool(enabled) and WANDB_AVAILABLE

    def _global_step(self):
        try:
            return int(self._get_global_step())
        except Exception:
            return 0

    def _get_git_hash(self):
        repo_root = Path(__file__).resolve().parents[2]
        try:
            output = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return output.strip()
        except Exception:
            return None

    def write_metadata(self):
        """Write environment and configuration metadata to a JSON file (rank-0 only)."""
        if not self.enabled:
            return
        if self.metadata_written or self.global_rank != 0:
            return
        logging_dir = self.config.get("logging_dir", ".")
        os.makedirs(logging_dir, exist_ok=True)
        env_keys = [
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "OMP_NUM_THREADS",
        ]
        metadata = {
            "git_hash": self._get_git_hash(),
            "config_yaml": yaml.safe_dump(self.config, sort_keys=False),
            "hostname": socket.gethostname(),
            "world_size": self.world_size,
            "env": {key: os.environ.get(key) for key in env_keys if key in os.environ},
            "torch_version": torch.__version__,
            "dist_backend": dist.get_backend() if dist.is_initialized() else None,
        }
        metadata_path = os.path.join(logging_dir, "throughput_metadata.json")
        with open(metadata_path, "w") as handle:
            json.dump(metadata, handle, indent=2)
        self.metadata_written = True
        if self.wandb_enabled:
            wandb.log({"throughput/metadata_path": metadata_path}, step=self._global_step())

    def maybe_start_measurement(self):
        """Begin measurement if warmup is complete and measurement has not yet run.

        Returns:
            bool: ``True`` if measurement is now active, ``False`` otherwise.
        """
        if not self.enabled or self.has_run:
            return False
        if self.measure_started:
            return True
        if self.step_index < self.warmup_steps:
            return False
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            dist.barrier()
        self.measure_started = True
        self.measure_count = 0
        if self.global_rank == 0:
            print(
                f"Starting throughput measurement: warmup={self.warmup_steps}, "
                f"measure={self.measure_steps}"
            )
        self.write_metadata()
        return True

    def measure_active(self):
        """Return whether throughput measurement is currently in progress.

        Returns:
            bool: ``True`` when actively measuring.
        """
        return self.measure_started and not self.has_run

    def accelerator_synchronize(self):
        """Synchronize the current accelerator device for accurate timing."""
        if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
            torch.accelerator.synchronize()

    def get_batch_samples(self, batch):
        """Determine the number of samples in a batch.

        Args:
            batch: PyG batch object or plain tensor.

        Returns:
            int: Number of samples in the batch.
        """
        if hasattr(batch, "num_graphs"):
            return int(batch.num_graphs)
        if torch.is_tensor(batch):
            return int(batch.size(0))
        return int(self.loader_config.get("batch_size", 1))

    def record_step(self, step_metrics):
        """Record a single step's throughput metrics.

        Args:
            step_metrics (dict): Metrics for the step, must include
                ``'throughput/samples_per_sec'``.
        """
        self.samples.append(step_metrics)
        if self.wandb_enabled:
            wandb.log(step_metrics, step=self._global_step())

    def on_step_end(self, step_metrics=None):
        """Called at the end of each training step to record metrics and check completion.

        Automatically calls ``finalize`` once the measurement window is filled.

        Args:
            step_metrics (dict, optional): Throughput metrics for this step.
        """
        if not self.enabled or self.has_run:
            return
        self.step_index += 1
        if not self.measure_active():
            return
        if step_metrics is None:
            return
        self.record_step(step_metrics)
        self.measure_count += 1
        if self.measure_count >= self.measure_steps:
            self.finalize()

    def finalize(self, partial=False):
        """Compute and log the final throughput summary across all ranks.

        Gathers per-step samples/sec from all DDP ranks, computes the global
        mean, and logs to W&B if enabled.

        Args:
            partial (bool): If ``True``, indicates the measurement window
                was not fully completed.
        """
        if not self.measure_started:
            return
        if self.has_run:
            return
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            dist.barrier()
        if not self.samples:
            if self.global_rank == 0:
                print("Throughput measurement skipped: no samples collected.")
            self.has_run = True
            return

        samples_per_sec = [sample["throughput/samples_per_sec"] for sample in self.samples]
        global_samples_per_sec = samples_per_sec
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, samples_per_sec)
            global_samples_per_sec = [value for sublist in gathered for value in sublist]

        sample_array = np.array(global_samples_per_sec)
        global_mean = float(sample_array.mean())

        summary = {
            "throughput/summary/mean_samples_per_sec": global_mean,
            "throughput/summary/partial": float(partial),
        }

        if self.global_rank == 0:
            status = "partial" if partial else "complete"
            print(f"Throughput measurement {status}: mean_samples_per_sec={global_mean:.3f}")
        if self.wandb_enabled:
            wandb.log(summary, step=self._global_step())

        self.has_run = True
