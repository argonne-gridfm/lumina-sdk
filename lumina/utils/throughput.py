import csv
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
    def __init__(
        self,
        config,
        world_size,
        global_rank,
        get_global_step,
        wandb_enabled=False,
        run_name=None,
        model_num_parameters=None,
        model_type=None,
        loss_type=None,
    ):
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
        self.logging_dir = self.config.get("logging_dir") or self.config.get("checkpoint_dir", ".")
        safe_chars = [
            ch if ch.isascii() and (ch.isalnum() or ch in ("-", "_", ".")) else "_"
            for ch in str(run_name or "run")
        ]
        self.run_stem = "".join(safe_chars).strip("._") or "run"
        self.model_num_parameters = model_num_parameters
        self.model_type = model_type
        self.loss_type = loss_type

    def set_wandb_enabled(self, enabled):
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
        if not self.enabled:
            return
        if self.metadata_written or self.global_rank != 0:
            return
        logging_dir = self.logging_dir
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
            "model_num_parameters": self.model_num_parameters,
            "model_type": self.model_type,
            "loss_type": self.loss_type,
            "env": {key: os.environ.get(key) for key in env_keys if key in os.environ},
            "torch_version": torch.__version__,
            "torch_hip_version": getattr(torch.version, "hip", None),
            "accelerator_name": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else None
            ),
            "dist_backend": dist.get_backend() if dist.is_initialized() else None,
        }
        metadata_path = os.path.join(logging_dir, f"{self.run_stem}-throughput_metadata.json")
        with open(metadata_path, "w") as handle:
            json.dump(metadata, handle, indent=2)
        self.metadata_written = True
        if self.wandb_enabled:
            wandb.log({"throughput/metadata_path": metadata_path}, step=self._global_step())

    def maybe_start_measurement(self):
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
        self._reset_peak_memory()
        if self.global_rank == 0:
            print(
                f"Starting throughput measurement: warmup={self.warmup_steps}, "
                f"measure={self.measure_steps}"
            )
        self.write_metadata()
        return True

    def measure_active(self):
        return self.measure_started and not self.has_run

    def accelerator_synchronize(self):
        if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
            torch.accelerator.synchronize()

    def _accelerator_module(self):
        for name in ("xpu", "cuda"):
            module = getattr(torch, name, None)
            if module is not None and getattr(module, "is_available", lambda: False)():
                return module
        return None

    def _reset_peak_memory(self):
        module = self._accelerator_module()
        reset = getattr(module, "reset_peak_memory_stats", None) if module is not None else None
        if reset is not None:
            reset()

    def _peak_memory_metrics(self):
        module = self._accelerator_module()
        if module is None:
            return {}
        metrics = {}
        for name, key in (
            ("max_memory_allocated", "memory/peak_allocated_bytes"),
            ("max_memory_reserved", "memory/peak_reserved_bytes"),
        ):
            getter = getattr(module, name, None)
            if getter is not None:
                try:
                    metrics[key] = float(getter())
                except TypeError:
                    metrics[key] = float(getter(None))
        return metrics

    def get_batch_samples(self, batch):
        if hasattr(batch, "num_graphs"):
            return int(batch.num_graphs)
        if torch.is_tensor(batch):
            return int(batch.size(0))
        return int(self.loader_config.get("batch_size", 1))

    def record_step(self, step_metrics):
        self.samples.append(step_metrics)
        if self.wandb_enabled:
            wandb.log(step_metrics, step=self._global_step())

    def on_step_end(self, step_metrics=None):
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

        peak_memory = self._peak_memory_metrics()
        local_keys = sorted({key for sample in self.samples for key in sample})
        all_keys = list(local_keys)
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            gathered_keys = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_keys, local_keys)
            all_keys = sorted({key for keys in gathered_keys for key in keys})
        # Step metrics are numeric; using the shared key order keeps collectives
        # identical on every rank even if one rank has no local value for a key.
        numeric_keys = all_keys

        summary = {
            "throughput/summary/partial": float(partial),
        }
        for key in numeric_keys:
            local_values = [
                float(sample[key])
                for sample in self.samples
                if isinstance(sample.get(key), (int, float, np.number))
            ]
            totals = torch.tensor(
                [sum(local_values), len(local_values)],
                dtype=torch.float64,
                device=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
            )
            if dist.is_available() and dist.is_initialized() and self.world_size > 1:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            if totals[1].item() > 0:
                summary[f"summary/mean/{key}"] = float((totals[0] / totals[1]).item())
        summary["throughput/summary/mean_samples_per_sec"] = summary.get(
            "summary/mean/throughput/samples_per_sec", 0.0
        )
        memory_values = torch.tensor(
            [
                peak_memory.get("memory/peak_allocated_bytes", 0.0),
                peak_memory.get("memory/peak_reserved_bytes", 0.0),
            ],
            dtype=torch.float64,
            device=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
        )
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(memory_values, op=dist.ReduceOp.MAX)
        summary["memory/peak_allocated_bytes"] = float(memory_values[0].item())
        summary["memory/peak_reserved_bytes"] = float(memory_values[1].item())

        if self.global_rank == 0:
            os.makedirs(self.logging_dir, exist_ok=True)
            steps_path = os.path.join(self.logging_dir, f"{self.run_stem}-throughput_steps.csv")
            with open(steps_path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=all_keys)
                writer.writeheader()
                writer.writerows(
                    {key: sample.get(key) for key in all_keys}
                    for sample in self.samples
                )
            summary_path = os.path.join(self.logging_dir, f"{self.run_stem}-throughput_summary.json")
            with open(summary_path, "w") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
            status = "partial" if partial else "complete"
            mean_throughput = summary.get("summary/mean/throughput/samples_per_sec", 0.0)
            print(
                f"Throughput measurement {status}: mean_samples_per_sec={mean_throughput:.3f}; "
                f"summary={summary_path}"
            )
        if self.wandb_enabled:
            wandb.log(summary, step=self._global_step())

        self.has_run = True
