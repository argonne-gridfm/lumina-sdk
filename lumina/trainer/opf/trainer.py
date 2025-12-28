import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from lumina.dataset.opf.opf_dataset import OPFDataset, OPFMultiDataset
from lumina.model.base.utils import describe_model
from lumina.model.opf.hetero_model import HEAT, HGT, RGAT, OPFHeteroGNN
from lumina.model.opf.homo_model import get_gnnNets
from lumina.model.opf.losses import OPFLossManager
from lumina.trainer.opf.utils import HETERO_MODEL_TYPES, initialize_model
from lumina.utils.graph_utils import HomoOPFDataset, convert_opf_to_homo
from lumina.utils.throughput import ThroughputTracker


def _normalize_group_ids(group_ids):
    if group_ids is None:
        return []
    if isinstance(group_ids, int):
        return [group_ids]
    if isinstance(group_ids, str):
        return [int(group_ids)]
    return [int(group_id) for group_id in group_ids]


class BaseOPFTrainer:
    TRAIN_METRIC_NAMES = (
        "train/loss/total",
        "train/loss/objective",
        "train/loss/lagrange_term",
        "train/loss/penalty_term",
        "train/feas/total_violation",
        "train/feas/total_violation_ema",
        "train/feas/total_violation_norm",
        "train/feas/total_violation_ema_norm",
        "train/feas/p_balance_rmse_pu",
        "train/feas/q_balance_rmse_pu",
        "train/feas/line_limit_rmse_pu",
        "train/lagrangian/penalty_mu",
        "train/lagrangian/multiplier_norm",
    )
    VAL_METRIC_NAMES = (
        "val/loss/total",
        "val/loss/objective",
        "val/feas/total_violation",
        "val/feas/total_violation_norm",
        "val/feas/p_balance_rmse_pu",
        "val/feas/q_balance_rmse_pu",
        "val/feas/line_limit_rmse_pu",
    )
    TRAIN_METRIC_MAP = (
        ("objective", "train/loss/objective"),
        ("lagrange_term", "train/loss/lagrange_term"),
        ("penalty_term", "train/loss/penalty_term"),
        ("raw_constraint_violation", "train/feas/total_violation"),
        ("ema_constraint_violation", "train/feas/total_violation_ema"),
        ("raw_constraint_violation_norm", "train/feas/total_violation_norm"),
        ("ema_constraint_violation_norm", "train/feas/total_violation_ema_norm"),
        ("p_balance_rmse", "train/feas/p_balance_rmse_pu"),
        ("q_balance_rmse", "train/feas/q_balance_rmse_pu"),
        ("line_limit_rmse", "train/feas/line_limit_rmse_pu"),
        ("penalty_parameter", "train/lagrangian/penalty_mu"),
        ("last_multiplier_norm", "train/lagrangian/multiplier_norm"),
    )
    VAL_METRIC_MAP = (
        ("raw_constraint_violation", "val/feas/total_violation"),
        ("raw_constraint_violation_norm", "val/feas/total_violation_norm"),
        ("p_balance_rmse", "val/feas/p_balance_rmse_pu"),
        ("q_balance_rmse", "val/feas/q_balance_rmse_pu"),
        ("line_limit_rmse", "val/feas/line_limit_rmse_pu"),
    )
    TRAIN_METRIC_GROUPS = (
        (
            "train/loss/total",
            "train/loss/objective",
            "train/loss/lagrange_term",
            "train/loss/penalty_term",
        ),
        (
            "train/feas/total_violation",
            "train/feas/total_violation_ema",
            "train/feas/total_violation_norm",
            "train/feas/total_violation_ema_norm",
            "train/feas/p_balance_rmse_pu",
            "train/feas/q_balance_rmse_pu",
            "train/feas/line_limit_rmse_pu",
        ),
        ("train/lagrangian/penalty_mu", "train/lagrangian/multiplier_norm"),
    )
    VAL_METRIC_GROUPS = (
        ("val/loss/total", "val/loss/objective", "val/score"),
        (
            "val/feas/total_violation",
            "val/feas/total_violation_norm",
            "val/feas/p_balance_rmse_pu",
            "val/feas/q_balance_rmse_pu",
            "val/feas/line_limit_rmse_pu",
        ),
    )

    def __init__(
        self,
        config,
        model_type,
        loss_type="mse",
        minmax_scaling=True,
        local_rank=0,
        global_rank=0,
        world_size=1,
        wandb_run_name=None,
        wandb_group_name=None,
        wandb_requested=False,
        wandb_project="lumina-training",
        wandb_entity=None,
        run_metadata=None,
    ):
        self.config = config
        self.model_type = model_type
        self.loss_type = loss_type
        self.minmax_scaling = minmax_scaling
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        self.wandb_run_name = wandb_run_name
        self.wandb_group_name = wandb_group_name
        self.wandb_requested = wandb_requested
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.run_metadata = run_metadata
        self.model_summary = None

        training_config = self.config["training"]
        self.max_epochs = training_config["max_epochs"]
        self.patience = training_config["patience"]
        self.grad_clip_val = training_config.get("gradient_clip_val")
        self.grad_clip_algo = training_config.get("gradient_clip_algorithm", "norm")
        self.accumulate_grad_batches = max(1, int(training_config.get("accumulate_grad_batches", 1)))
        self.log_every_n_steps = training_config.get("log_every_n_steps", 0)
        self.log_every_n_samples = int(training_config.get("log_every_n_samples", 512) or 0)
        val_every_n_epochs = training_config.get(
            "val_every_n_epochs",
            training_config.get("val_check_interval", 1),
        )
        self.val_every_n_epochs = max(1, int(val_every_n_epochs or 1))
        self.val_every_n_samples = int(training_config.get("val_every_n_samples") or 0)
        self.max_global_samples = int(training_config.get("max_global_samples") or 0)
        score_alpha = training_config.get("score_alpha", 1.0)
        self.score_alpha = float(1.0 if score_alpha is None else score_alpha)
        self.log_normalized_violation = bool(training_config.get("log_normalized_violation", False))

        checkpoint_config = self.config.get("checkpointing", {})
        self.ckpt_every_n_epochs = int(checkpoint_config.get("every_n_epochs") or 0)
        self.ckpt_every_n_samples = int(checkpoint_config.get("every_n_samples") or 0)
        self.save_last_checkpoint = bool(checkpoint_config.get("save_last", False))

        self.checkpoint_dir = config["checkpoint_dir"]

        self._load_data()
        self._create_dataloaders()
        self._init_sample_schedules()
        self._create_model()
        self._initialize_loss_managers()
        self._init_optimizer()

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.global_samples = 0
        self.stop_training = False
        self._next_log_samples = self.log_every_n_samples if self.log_every_n_samples > 0 else None

        self.wandb_run = None
        self.wandb_enabled = False

        self.train_metric_names = list(self.TRAIN_METRIC_NAMES)
        self.val_metric_names = list(self.VAL_METRIC_NAMES)
        self.train_metric_map = list(self.TRAIN_METRIC_MAP)
        self.val_metric_map = list(self.VAL_METRIC_MAP)
        self.train_metric_groups = [list(group) for group in self.TRAIN_METRIC_GROUPS]
        self.val_metric_groups = [list(group) for group in self.VAL_METRIC_GROUPS]

        self._init_wandb()
        self.throughput_tracker = None
        if training_config.get("throughput_enabled", True):
            self.throughput_tracker = ThroughputTracker(
                config=self.config,
                world_size=self.world_size,
                global_rank=self.global_rank,
                get_global_step=lambda: self.global_samples,
                wandb_enabled=self.wandb_enabled,
            )
            if not self.throughput_tracker.enabled:
                self.throughput_tracker = None

    def _load_data(self):
        raise NotImplementedError

    def _create_dataloaders(self):
        raise NotImplementedError

    def _create_model(self):
        raise NotImplementedError

    def _initialize_loss_managers(self):
        raise NotImplementedError

    def _default_wandb_run_name(self):
        return f"acopf-ddp-{self.model_type}-{self.loss_type}"

    def _should_print_epoch(self):
        return self.global_rank == 0 and not self.wandb_enabled

    def _checkpoint_tag(self):
        raise NotImplementedError

    def _checkpoint_payload(self):
        return {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

    def _on_checkpoint_saved(self, filepath):
        return

    def _build_model(self, sample_data, metadata, per_node_output_size):
        if self.global_rank == 0:
            print(f"Per-node output size: {per_node_output_size}")

        if self.model_type in HETERO_MODEL_TYPES:
            input_channels = {}
            node_types = list(metadata["nodes"].keys())
            edge_types = list(metadata["edges"].keys())
            metadata_tuple = (node_types, edge_types)

            for node_type in node_types:
                if node_type in sample_data.x_dict:
                    input_channels[node_type] = sample_data[node_type].x.shape[1]

            if self.model_type in self.config["models"]:
                model_config = self.config["models"][self.model_type]
            else:
                if self.global_rank == 0:
                    print(f"Warning: Config for {self.model_type} not found, using HeteroGNN config")
                model_config = self.config["models"]["HeteroGNN"]

            if self.model_type == "HeteroGNN":
                model_class = OPFHeteroGNN
            elif self.model_type == "RGAT":
                model_class = RGAT
            elif self.model_type == "HEAT":
                model_class = HEAT
            elif self.model_type == "HGT":
                model_class = HGT

            kwargs = {
                "metadata": metadata_tuple,
                "input_channels": input_channels,
                "hidden_channels": model_config["hidden_channels"],
                "out_channels": per_node_output_size,
                "num_layers": model_config["num_layers"],
                "backend": model_config.get("backend", "sage"),
            }

            if self.model_type in {"RGAT", "HGT"}:
                kwargs["num_heads"] = model_config.get("num_heads", 1)
            if self.model_type == "HEAT":
                kwargs["attention_heads"] = model_config.get("attention_heads", 1)

            model = model_class(**kwargs)
            initialize_model(model, sample_data, self.device)

            if self.global_rank == 0:
                print(f"{self.model_type} Model created")
                self.model_summary = describe_model(
                    model,
                    model_type=self.model_type,
                    model_config=model_config,
                    print_fn=print,
                )
            else:
                self.model_summary = None
        else:
            homo_sample = self._get_homo_sample(sample_data)
            input_dim = homo_sample.x.shape[1]

            if self.model_type in self.config["models"]:
                model_config = self.config["models"][self.model_type]
            elif "HomoGNN" in self.config["models"]:
                model_config = self.config["models"]["HomoGNN"]
            else:
                model_config = {
                    "hidden_dim": 64,
                    "num_layers": 3,
                    "dropout": 0.1,
                    "readout": "mean",
                    "edge_dim": homo_sample.edge_attr.shape[1],
                }

            model_config["model_name"] = self.model_type
            if "edge_dim" not in model_config:
                edge_attr = getattr(homo_sample, "edge_attr", None)
                if edge_attr is None:
                    model_config["edge_dim"] = 1
                else:
                    edge_dim = edge_attr.size(-1) if edge_attr.dim() > 1 else 1
                    model_config["edge_dim"] = int(edge_dim)

            model = get_gnnNets(
                input_dim=input_dim,
                output_dim=per_node_output_size,
                model_params=model_config,
            )

            initialize_model(model, homo_sample, self.device)
            if self.global_rank == 0:
                print(f"{self.model_type} Model created")
                self.model_summary = describe_model(
                    model,
                    model_type=self.model_type,
                    model_config=model_config,
                    print_fn=print,
                )
            else:
                self.model_summary = None

        model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=True)
        return model

    def _get_homo_sample(self, sample_data):
        if hasattr(self, "train_loader") and self.train_loader is not None:
            try:
                return self.train_loader.dataset[0]
            except Exception:
                pass
        if hasattr(self, "train_loaders") and self.train_loaders:
            for loader in self.train_loaders.values():
                if loader is None:
                    continue
                try:
                    return loader.dataset[0]
                except Exception:
                    continue
        return convert_opf_to_homo(sample_data)

    def _init_optimizer(self):
        optimizer_config = self.config["optimizer"]
        if "Adam" in optimizer_config:
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_config["Adam"])
        elif "AdamW" in optimizer_config:
            self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_config["AdamW"])

    def _clip_gradients(self):
        if self.grad_clip_val is None or self.grad_clip_val <= 0:
            return

        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.grad_clip_algo == "value":
            torch.nn.utils.clip_grad_value_(parameters, self.grad_clip_val)
        else:
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip_val)

    def _init_wandb(self):
        if self.wandb_requested is False:
            return
        if not WANDB_AVAILABLE:
            if self.global_rank == 0 and self.wandb_requested:
                print("Warning: Weights & Biases is not available. Install wandb or omit --wandb.")
            return
        if self.global_rank != 0:
            return
        if wandb.run is not None:
            self.wandb_run = wandb.run
            self.wandb_enabled = True
            try:
                wandb.config.update(self.config, allow_val_change=True)
            except Exception:
                pass
            if self.run_metadata:
                try:
                    wandb.config.update({"run_metadata": self.run_metadata}, allow_val_change=True)
                except Exception:
                    pass
            self._log_model_summary()
            return
        logging_dir = self.config.get("logging_dir")
        run_name = self.wandb_run_name or self._default_wandb_run_name()
        try:
            wandb_kwargs = {
                "project": self.wandb_project,
                "name": run_name,
                "dir": logging_dir,
                "config": self.config,
                "group": self.wandb_group_name,
            }
            if self.wandb_entity:
                wandb_kwargs["entity"] = self.wandb_entity
            self.wandb_run = wandb.init(**wandb_kwargs)
            self.wandb_enabled = True
            if self.run_metadata:
                try:
                    wandb.config.update({"run_metadata": self.run_metadata}, allow_val_change=True)
                except Exception:
                    pass
            self._log_model_summary()
        except Exception as exc:
            print(f"Warning: W&B init failed: {exc}")
            self.wandb_run = None
            self.wandb_enabled = False

    def _log_model_summary(self):
        if not self.model_summary or self.wandb_run is None:
            return
        try:
            self.wandb_run.summary["model_summary"] = self.model_summary
        except Exception:
            pass
        try:
            wandb.config.update({"model_summary": self.model_summary}, allow_val_change=True)
        except Exception:
            pass

    def _should_log_step(self):
        if not self.wandb_enabled:
            return False
        if self.log_every_n_samples and self.log_every_n_samples > 0:
            return self.global_samples >= self._next_log_samples
        if self.log_every_n_steps and self.log_every_n_steps > 0:
            return self.global_step % self.log_every_n_steps == 0
        return True

    def _log_wandb_step(self, loss_value, loss_info):
        if not self._should_log_step():
            return
        metrics = {
            "train/loss/total": self._as_float(loss_value),
            "train/samples_seen": int(self.global_samples),
        }
        for info_key, metric_name in self.train_metric_map:
            if info_key in loss_info:
                metric_value = self._as_float(loss_info[info_key])
                if metric_value is not None:
                    metrics[metric_name] = metric_value
        wandb.log(metrics, step=self.global_samples)
        if self._next_log_samples is not None and self.log_every_n_samples > 0:
            while self.global_samples >= self._next_log_samples:
                self._next_log_samples += self.log_every_n_samples

    def _log_wandb_validation(self, metric_avgs):
        if not self.wandb_enabled or not metric_avgs:
            return
        metrics = dict(metric_avgs)
        wandb.log(metrics, step=self.global_samples)

    def _as_float(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.detach().item()
            return value.detach().float().mean().item()
        if isinstance(value, np.ndarray):
            return float(value.mean())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _init_metric_trackers(self, metric_names):
        metric_sums = {name: 0.0 for name in metric_names}
        metric_counts = {name: 0.0 for name in metric_names}
        return metric_sums, metric_counts

    def _get_batch_samples(self, batch):
        if hasattr(batch, "num_graphs"):
            return int(batch.num_graphs)
        if torch.is_tensor(batch):
            return int(batch.size(0))
        if hasattr(batch, "x") and torch.is_tensor(batch.x):
            return int(batch.x.size(0))
        loader_config = self.config.get("loader", {})
        return int(loader_config.get("batch_size", 1))

    def _train_samples_per_epoch(self):
        if hasattr(self, "train_sampler") and self.train_sampler is not None:
            total_size = getattr(self.train_sampler, "total_size", None)
            if total_size is not None:
                return int(total_size)
            num_samples = getattr(self.train_sampler, "num_samples", None)
            if num_samples is not None:
                return int(num_samples) * self.world_size

        total = 0
        if hasattr(self, "train_samplers") and self.train_samplers:
            for sampler in self.train_samplers.values():
                if sampler is None:
                    continue
                total_size = getattr(sampler, "total_size", None)
                if total_size is not None:
                    total += int(total_size)
                else:
                    num_samples = getattr(sampler, "num_samples", None)
                    if num_samples is not None:
                        total += int(num_samples) * self.world_size
            if total > 0:
                return total

        if hasattr(self, "train_loader"):
            try:
                return int(len(self.train_loader.dataset))
            except Exception:
                pass
        if hasattr(self, "train_loaders"):
            for loader in self.train_loaders.values():
                try:
                    total += int(len(loader.dataset))
                except Exception:
                    continue
            if total > 0:
                return total
        return 0

    def _init_sample_schedules(self):
        samples_per_epoch = self._train_samples_per_epoch()
        if self.val_every_n_samples <= 0 and self.val_every_n_epochs > 0 and samples_per_epoch > 0:
            self.val_every_n_samples = int(self.val_every_n_epochs * samples_per_epoch)
        if self.val_every_n_samples > 0:
            self._next_val_samples = self.val_every_n_samples
        else:
            self._next_val_samples = None

        if self.ckpt_every_n_samples <= 0 and self.ckpt_every_n_epochs > 0 and samples_per_epoch > 0:
            self.ckpt_every_n_samples = int(self.ckpt_every_n_epochs * samples_per_epoch)
        if self.ckpt_every_n_samples > 0:
            self._next_ckpt_samples = self.ckpt_every_n_samples
        else:
            self._next_ckpt_samples = None

    def _update_global_samples(self, batch):
        batch_samples = self._get_batch_samples(batch)
        self.global_samples += batch_samples * self.world_size
        return batch_samples

    def _maybe_stop_by_samples(self):
        if self.max_global_samples <= 0:
            return False
        should_stop = self.global_samples >= self.max_global_samples
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            stop_tensor = torch.tensor(int(should_stop), device=self.device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            self.stop_training = True
        return should_stop

    def _sync_batch_samples(self, batch_samples):
        if not dist.is_available() or not dist.is_initialized() or self.world_size <= 1:
            return int(batch_samples)
        sample_tensor = torch.tensor(int(batch_samples), device=self.device)
        dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
        return int(sample_tensor.item())

    def _sync_trigger(self, should_run):
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            flag = torch.tensor(int(should_run), device=self.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            return bool(flag.item())
        return should_run

    def _advance_next_samples(self, next_samples, interval):
        if interval <= 0:
            return None
        if next_samples is None:
            next_samples = interval
        while next_samples <= self.global_samples:
            next_samples += interval
        return next_samples

    def _maybe_run_validation(self):
        if self.val_every_n_samples <= 0 or self._next_val_samples is None:
            return False
        should_run = self.global_samples >= self._next_val_samples
        should_run = self._sync_trigger(should_run)
        if not should_run:
            return False
        self._next_val_samples = self._advance_next_samples(
            self._next_val_samples,
            self.val_every_n_samples,
        )
        self._run_validation()
        return True

    def _run_validation(self):
        val_loss, val_task_loss, val_metrics = self.validate()
        if self.wandb_enabled:
            self._log_wandb_validation(val_metrics)
        if val_loss is None:
            self.model.train()
            return None, None, None

        if self._should_print_epoch():
            print(f"\nValidation @ samples {self.global_samples} (epoch {self.current_epoch}):")
            print(f"  Val Loss: {val_loss:.4f}, Val Task: {val_task_loss:.4f}")
            if val_metrics:
                self._print_metric_groups("  Val Metrics:", val_metrics, self.val_metric_groups)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"best-{self._checkpoint_tag()}-epoch{self.current_epoch:02d}-val{val_loss:.4f}.pt",
            )
            self.save_checkpoint(checkpoint_path)
        else:
            self.patience_counter += 1
            if self.global_rank == 0:
                print(f"  No improvement. Patience: {self.patience_counter}/{self.patience}")
            if self.patience_counter >= self.patience:
                if self.global_rank == 0:
                    print(f"\nEarly stopping triggered after {self.current_epoch + 1} epochs")
                self.stop_training = True

        self.model.train()
        return val_loss, val_task_loss, val_metrics

    def _maybe_save_periodic_checkpoint(self):
        if self.ckpt_every_n_samples <= 0 or self._next_ckpt_samples is None:
            return False
        should_save = self.global_samples >= self._next_ckpt_samples
        should_save = self._sync_trigger(should_save)
        if not should_save:
            return False
        self._next_ckpt_samples = self._advance_next_samples(
            self._next_ckpt_samples,
            self.ckpt_every_n_samples,
        )
        self._save_periodic_checkpoint()
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            dist.barrier()
        return True

    def _save_periodic_checkpoint(self):
        if self.global_rank != 0:
            return
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint-{self._checkpoint_tag()}-samples{self.global_samples}.pt",
        )
        self.save_checkpoint(checkpoint_path)

    def _sync_lagrangian_inputs(self, constraint_violation, constraints, batch_samples=None, total_batch_samples=None):
        if not dist.is_available() or not dist.is_initialized() or self.world_size <= 1:
            return constraint_violation, constraints

        synced_constraints = constraints
        synced_violation = constraint_violation
        weight = None
        total_weight = float(self.world_size)
        if batch_samples is not None and total_batch_samples:
            weight = float(batch_samples)
            total_weight = float(total_batch_samples)

        with torch.no_grad():
            if torch.is_tensor(synced_constraints) and synced_constraints.numel() > 0:
                weighted_constraints = synced_constraints.detach()
                if weight is not None:
                    weighted_constraints = weighted_constraints * weight
                dist.all_reduce(weighted_constraints, op=dist.ReduceOp.SUM)
                synced_constraints = weighted_constraints / total_weight

            if synced_violation is not None:
                if torch.is_tensor(synced_violation):
                    weighted_violation = synced_violation.detach()
                else:
                    weighted_violation = torch.tensor(float(synced_violation), device=self.device)
                if weight is not None:
                    weighted_violation = weighted_violation * weight
                dist.all_reduce(weighted_violation, op=dist.ReduceOp.SUM)
                synced_violation = weighted_violation / total_weight

        return synced_violation, synced_constraints

    def _add_metric(self, metric_sums, metric_counts, name, value, weight=1.0):
        numeric_value = self._as_float(value)
        if numeric_value is None:
            return
        metric_sums[name] += numeric_value * weight
        metric_counts[name] += weight

    def _reduce_metrics(self, metric_sums, metric_counts):
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            for name in metric_sums:
                sum_tensor = torch.tensor(metric_sums[name], device=self.device)
                count_tensor = torch.tensor(metric_counts[name], device=self.device)
                dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                metric_sums[name] = sum_tensor.item()
                metric_counts[name] = count_tensor.item()
        return metric_sums, metric_counts

    def _compute_metric_avgs(self, metric_sums, metric_counts):
        metric_avgs = {}
        for name, total in metric_sums.items():
            count = metric_counts.get(name, 0.0)
            if count > 0:
                metric_avgs[name] = total / count
        return metric_avgs

    def _add_val_score(self, metric_avgs):
        if not metric_avgs:
            return
        objective = metric_avgs.get("val/loss/objective")
        if objective is None:
            objective = metric_avgs.get("val/loss/total")
        violation = metric_avgs.get("val/feas/total_violation")
        if objective is None or violation is None:
            return
        metric_avgs["val/score"] = objective + self.score_alpha * violation

    def _print_metric_groups(self, title, metric_avgs, groups):
        if not metric_avgs:
            return
        print(title)
        for names in groups:
            parts = []
            for name in names:
                if name in metric_avgs:
                    parts.append(f"{name}: {metric_avgs[name]:.4f}")
            if parts:
                print("    " + ", ".join(parts))

    def forward(self, batch):
        if self.model_type in HETERO_MODEL_TYPES:
            x_dict = {k: v.float() for k, v in batch.x_dict.items()}
            return self.model.module(x_dict, batch.edge_index_dict, minmax_scaling=self.minmax_scaling)
        if isinstance(batch, torch.Tensor) or hasattr(batch, "node_type"):
            homo_batch = batch
        else:
            homo_batch = convert_opf_to_homo(batch)
            homo_batch = homo_batch.to(self.device)

        if hasattr(homo_batch, "x"):
            homo_batch.x = homo_batch.x.float()
        if hasattr(homo_batch, "edge_attr") and homo_batch.edge_attr is not None:
            homo_batch.edge_attr = homo_batch.edge_attr.float()

        homo_output = self.model.module(homo_batch)

        predictions = {}
        node_types = ["bus", "generator", "load", "shunt"]
        for i, node_type in enumerate(node_types):
            mask = homo_batch.node_type == i
            if mask.any():
                predictions[node_type] = homo_output[mask]
        return predictions

    def save_checkpoint(self, filepath):
        if self.global_rank == 0:
            checkpoint = self._checkpoint_payload()
            torch.save(checkpoint, filepath)
            print(f"Checkpoint saved to {filepath}")
            self._on_checkpoint_saved(filepath)

    def train(self):
        checkpoint_dir = self.checkpoint_dir
        if self.global_rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
        if self.throughput_tracker:
            self.throughput_tracker.write_metadata()

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            train_loss, train_task_loss, train_metrics = self.train_epoch(epoch)

            if self._should_print_epoch():
                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Task: {train_task_loss:.4f}")
                self._print_metric_groups("  Train Metrics:", train_metrics, self.train_metric_groups)
            if self.stop_training:
                if self.max_global_samples > 0 and self.global_samples >= self.max_global_samples:
                    if self.global_rank == 0:
                        print(
                            f"\nStopping after reaching max_global_samples={self.max_global_samples} "
                            f"(global_samples={self.global_samples})"
                        )
                dist.barrier()
                break

            dist.barrier()

        if self.throughput_tracker:
            self.throughput_tracker.finalize(partial=True)

        if self.wandb_enabled and wandb is not None:
            wandb.finish()


class OPFTrainer(BaseOPFTrainer):
    def __init__(
        self,
        config,
        case_name,
        group_ids,
        model_type,
        loss_type="mse",
        minmax_scaling=True,
        local_rank=0,
        global_rank=0,
        world_size=1,
        wandb_run_name=None,
        wandb_group_name=None,
        wandb_requested=False,
        wandb_project="lumina-training",
        wandb_entity=None,
        run_metadata=None,
    ):
        self.case_name = case_name
        self.group_ids = _normalize_group_ids(group_ids)
        if not self.group_ids:
            raise ValueError("group_ids must contain at least one group id.")
        super().__init__(
            config=config,
            model_type=model_type,
            loss_type=loss_type,
            minmax_scaling=minmax_scaling,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            wandb_run_name=wandb_run_name,
            wandb_group_name=wandb_group_name,
            wandb_requested=wandb_requested,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            run_metadata=run_metadata,
        )

    def _load_data(self):
        dataset_kwargs = dict(
            root=self.config["root"],
            case_name=self.case_name,
            local_raw_folder=self.config.get("local_raw_folder"),
            force_reload=False,
        )

        def build_dataset():
            if len(self.group_ids) == 1:
                return OPFDataset(group_id=self.group_ids[0], **dataset_kwargs)
            return OPFMultiDataset.from_case_groups(group_ids=self.group_ids, **dataset_kwargs)

        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            if self.global_rank == 0:
                self.dataset = build_dataset()
            dist.barrier()
            if self.global_rank != 0:
                self.dataset = build_dataset()
        else:
            self.dataset = build_dataset()
        if self.global_rank == 0:
            print(f"Dataset loaded: {len(self.dataset)} samples")

    def _create_dataloaders(self):
        n_samples = len(self.dataset)
        n_train = int(n_samples * self.config["train_split"])
        n_val = int(n_samples * self.config["val_split"])

        train_dataset = torch.utils.data.Subset(self.dataset, range(n_train))
        val_dataset = torch.utils.data.Subset(self.dataset, range(n_train, n_train + n_val))
        test_dataset = torch.utils.data.Subset(self.dataset, range(n_train + n_val, n_samples))

        if self.model_type not in HETERO_MODEL_TYPES:
            train_dataset = HomoOPFDataset(train_dataset)
            val_dataset = HomoOPFDataset(val_dataset)
            test_dataset = HomoOPFDataset(test_dataset)

        loader_config = self.config["loader"]

        self.train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=loader_config["shuffle"],
        )

        self.val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=False,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=loader_config["batch_size"],
            sampler=self.train_sampler,
            num_workers=loader_config["num_workers"],
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=loader_config["batch_size"],
            sampler=self.val_sampler,
            num_workers=loader_config["num_workers"],
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=loader_config["batch_size"],
            shuffle=False,
            num_workers=loader_config["num_workers"],
            pin_memory=True,
        )

    def _create_model(self):
        metadata = self.dataset.metadata()
        sample_data = self.dataset[0]
        per_node_output_size = sample_data["bus"].y.shape[-1]
        self.model = self._build_model(sample_data, metadata, per_node_output_size)

    def _initialize_loss_managers(self):
        lagrangian_config = self.config.get("lagrangian", {})
        self.loss_manager = OPFLossManager(
            loss_type=self.loss_type,
            device=self.device,
            lagrangian_config=lagrangian_config,
            log_normalized_violation=self.log_normalized_violation,
        )

        if self.global_rank == 0:
            print(f"Loss Manager initialized with loss_type='{self.loss_type}'")

    def _checkpoint_tag(self):
        return self.case_name

    def _on_checkpoint_saved(self, filepath):
        if self.wandb_enabled and self.wandb_run is not None:
            try:
                self.wandb_run.summary["best_model_path"] = filepath
            except Exception:
                pass

    def train_epoch(self, epoch):
        self.model.train()
        self.train_sampler.set_epoch(epoch)

        total_loss = 0.0
        total_task_loss = 0.0
        num_batches = 0
        total_steps = len(self.train_loader)
        metric_sums, metric_counts = self._init_metric_trackers(self.train_metric_names)
        step_start_time = None
        step_samples = 0
        accum_batches = 0
        tracker = self.throughput_tracker

        if self.global_rank == 0 and not self.wandb_enabled:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        else:
            pbar = self.train_loader

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            is_step_start = accum_batches == 0
            if is_step_start:
                if tracker:
                    tracker.maybe_start_measurement()
                    if tracker.measure_active():
                        tracker.accelerator_synchronize()
                        step_start_time = time.perf_counter()
                        step_samples = 0

            batch = batch.to(self.device)
            batch_samples = self._update_global_samples(batch)

            if tracker and tracker.measure_active():
                step_samples += tracker.get_batch_samples(batch)

            predictions = self.forward(batch)
            collect_constraints = self.loss_manager.lagrangian is not None
            loss, loss_info = self.loss_manager.compute_loss(
                predictions,
                batch,
                return_info=True,
                collect_constraints=collect_constraints,
            )

            global_batch_samples = batch_samples
            synced_violation = loss_info.get("constraint_violation")
            synced_constraints = loss_info.get("constraints")
            if self.loss_manager.lagrangian is not None:
                global_batch_samples = self._sync_batch_samples(batch_samples)
                synced_violation, synced_constraints = self._sync_lagrangian_inputs(
                    synced_violation,
                    synced_constraints,
                    batch_samples=batch_samples,
                    total_batch_samples=global_batch_samples,
                )

            self.loss_manager.update_lagrangian(
                constraint_violation=synced_violation,
                constraints=synced_constraints,
                is_training=self.model.training,
                sample_count=global_batch_samples,
            )

            loss_value = loss.item()
            self._add_metric(metric_sums, metric_counts, "train/loss/total", loss_value)
            if self.log_every_n_samples and self.log_every_n_samples > 0:
                self._log_wandb_step(loss_value, loss_info)
            loss = loss / self.accumulate_grad_batches

            loss.backward()

            should_step = ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (
                (batch_idx + 1) == total_steps
            )

            if should_step:
                if tracker and tracker.measure_active():
                    tracker.accelerator_synchronize()
                self._clip_gradients()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                if not self.log_every_n_samples or self.log_every_n_samples <= 0:
                    self._log_wandb_step(loss_value, loss_info)
                if tracker:
                    step_metrics = None
                    if tracker.measure_active():
                        tracker.accelerator_synchronize()
                        step_time = time.perf_counter() - step_start_time
                        total_samples = step_samples * self.world_size
                        samples_per_sec = total_samples / step_time if step_time > 0 else 0.0
                        step_metrics = {
                            "throughput/samples_per_sec": samples_per_sec,
                        }
                    tracker.on_step_end(step_metrics)
                accum_batches = 0
            else:
                accum_batches += 1

            total_loss += loss_value
            if "objective" in loss_info:
                objective_value = self._as_float(loss_info["objective"])
                if objective_value is not None:
                    total_task_loss += objective_value
                    self._add_metric(metric_sums, metric_counts, "train/loss/objective", objective_value)
            for info_key, metric_name in self.train_metric_map:
                if info_key in loss_info:
                    self._add_metric(metric_sums, metric_counts, metric_name, loss_info[info_key])
            num_batches += 1

            if self.global_rank == 0 and not self.wandb_enabled:
                if self.log_every_n_steps and self.log_every_n_steps > 0:
                    if self.global_step % self.log_every_n_steps == 0:
                        pbar.set_postfix({"loss": loss_value})
                else:
                    pbar.set_postfix({"loss": loss_value})

            if self._maybe_run_validation() and self.stop_training:
                break
            self._maybe_save_periodic_checkpoint()
            if self._maybe_stop_by_samples():
                break

        avg_loss = total_loss / num_batches
        avg_task_loss = total_task_loss / num_batches

        loss_tensor = torch.tensor([avg_loss, avg_task_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

        metric_sums, metric_counts = self._reduce_metrics(metric_sums, metric_counts)
        metric_avgs = self._compute_metric_avgs(metric_sums, metric_counts)

        self.loss_manager.step_epoch()

        return loss_tensor[0].item(), loss_tensor[1].item(), metric_avgs

    def validate(self):
        self.model.eval()

        total_loss = 0.0
        total_task_loss = 0.0
        num_batches = 0
        metric_sums, metric_counts = self._init_metric_trackers(self.val_metric_names)

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                predictions = self.forward(batch)
                loss, loss_info = self.loss_manager.compute_loss(predictions, batch, return_info=True)

                loss_value = loss.item()
                total_loss += loss_value
                self._add_metric(metric_sums, metric_counts, "val/loss/total", loss_value)
                if "objective" in loss_info:
                    objective_value = self._as_float(loss_info["objective"])
                    if objective_value is not None:
                        total_task_loss += objective_value
                        self._add_metric(metric_sums, metric_counts, "val/loss/objective", objective_value)
                for info_key, metric_name in self.val_metric_map:
                    if info_key in loss_info:
                        self._add_metric(metric_sums, metric_counts, metric_name, loss_info[info_key])
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_task_loss = total_task_loss / num_batches

        loss_tensor = torch.tensor([avg_loss, avg_task_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

        metric_sums, metric_counts = self._reduce_metrics(metric_sums, metric_counts)
        metric_avgs = self._compute_metric_avgs(metric_sums, metric_counts)
        self._add_val_score(metric_avgs)

        return loss_tensor[0].item(), loss_tensor[1].item(), metric_avgs


class MultiCaseOPFTrainer(BaseOPFTrainer):
    def __init__(
        self,
        config,
        case_names,
        group_ids,
        model_type,
        loss_type="mse",
        minmax_scaling=True,
        local_rank=0,
        global_rank=0,
        world_size=1,
        wandb_run_name=None,
        wandb_group_name=None,
        wandb_requested=False,
        wandb_project="lumina-training",
        wandb_entity=None,
        run_metadata=None,
    ):
        self.case_names = list(case_names)
        self.case_keys = {idx: f"case_{idx}" for idx in range(len(self.case_names))}
        self.group_ids = _normalize_group_ids(group_ids)
        if not self.group_ids:
            raise ValueError("group_ids must contain at least one group id.")
        super().__init__(
            config=config,
            model_type=model_type,
            loss_type=loss_type,
            minmax_scaling=minmax_scaling,
            local_rank=local_rank,
            global_rank=global_rank,
            world_size=world_size,
            wandb_run_name=wandb_run_name,
            wandb_group_name=wandb_group_name,
            wandb_requested=wandb_requested,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            run_metadata=run_metadata,
        )

    def _load_dataset(self, case_name):
        dataset_kwargs = dict(
            root=self.config["root"],
            case_name=case_name,
            local_raw_folder=self.config.get("local_raw_folder"),
            force_reload=False,
        )
        def build_dataset():
            if len(self.group_ids) == 1:
                return OPFDataset(group_id=self.group_ids[0], **dataset_kwargs)
            return OPFMultiDataset.from_case_groups(group_ids=self.group_ids, **dataset_kwargs)
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            if self.global_rank == 0:
                dataset = build_dataset()
            dist.barrier()
            if self.global_rank != 0:
                dataset = build_dataset()
        else:
            dataset = build_dataset()
        return dataset

    def _load_data(self):
        self.case_datasets = []
        reference_metadata = None
        reference_out_dim = None

        for case_name in self.case_names:
            dataset = self._load_dataset(case_name)
            if len(dataset) == 0:
                raise ValueError(f"Dataset for case {case_name} is empty.")

            if self.global_rank == 0:
                print(f"Dataset loaded for {case_name}: {len(dataset)} samples")

            metadata = dataset.metadata()
            sample = dataset[0]
            out_dim = sample["bus"].y.shape[-1]

            if reference_metadata is None:
                reference_metadata = metadata
                reference_out_dim = out_dim
            else:
                if metadata != reference_metadata:
                    raise ValueError(
                        f"Dataset metadata mismatch between cases. {case_name} does not share the same schema."
                    )
                if out_dim != reference_out_dim:
                    raise ValueError(
                        f"Output dimension mismatch for case {case_name} "
                        f"(expected {reference_out_dim}, found {out_dim})."
                    )

            self.case_datasets.append(dataset)

        self.reference_metadata = reference_metadata
        self.reference_output_dim = reference_out_dim

    def _create_dataloaders(self):
        self.train_loaders = {}
        self.val_loaders = {}
        self.test_loaders = {}
        self.train_samplers = {}
        self.val_samplers = {}
        self.train_case_indices = []
        self.val_case_indices = []
        self.test_case_indices = []

        loader_config = self.config["loader"]
        train_ratio = self.config["train_split"]
        val_ratio = self.config["val_split"]
        split_seed = int(self.config.get("split_seed", 42))

        for case_idx, dataset in enumerate(self.case_datasets):
            n_samples = len(dataset)
            train_len = max(1, int(n_samples * train_ratio))
            val_len = int(n_samples * val_ratio)
            if train_len + val_len >= n_samples:
                val_len = max(0, n_samples - train_len - 1)
            test_len = n_samples - train_len - val_len

            generator = torch.Generator().manual_seed(split_seed + case_idx)
            subsets = torch.utils.data.random_split(dataset, [train_len, val_len, test_len], generator=generator)
            train_dataset, val_dataset, test_dataset = subsets

            if self.model_type not in HETERO_MODEL_TYPES:
                train_dataset = HomoOPFDataset(train_dataset)
                if val_len > 0:
                    val_dataset = HomoOPFDataset(val_dataset)
                if test_len > 0:
                    test_dataset = HomoOPFDataset(test_dataset)

            self.train_samplers[case_idx] = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=loader_config["shuffle"],
            )
            self.train_loaders[case_idx] = DataLoader(
                train_dataset,
                batch_size=loader_config["batch_size"],
                sampler=self.train_samplers[case_idx],
                num_workers=loader_config["num_workers"],
                pin_memory=True,
            )
            self.train_case_indices.append(case_idx)

            if val_len > 0:
                self.val_samplers[case_idx] = DistributedSampler(
                    val_dataset,
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                    shuffle=False,
                )
                self.val_loaders[case_idx] = DataLoader(
                    val_dataset,
                    batch_size=loader_config["batch_size"],
                    sampler=self.val_samplers[case_idx],
                    num_workers=loader_config["num_workers"],
                    pin_memory=True,
                )
                self.val_case_indices.append(case_idx)

            if test_len > 0:
                self.test_loaders[case_idx] = DataLoader(
                    test_dataset,
                    batch_size=loader_config["batch_size"],
                    shuffle=False,
                    num_workers=loader_config["num_workers"],
                    pin_memory=True,
                )
                self.test_case_indices.append(case_idx)

    def _create_model(self):
        sample_data = self.case_datasets[0][0]
        per_node_output_size = self.reference_output_dim
        self.model = self._build_model(sample_data, self.reference_metadata, per_node_output_size)

    def _initialize_loss_managers(self):
        lagrangian_config = self.config.get("lagrangian", {})
        self.loss_managers = {}

        for case_idx in range(len(self.case_names)):
            self.loss_managers[case_idx] = OPFLossManager(
                loss_type=self.loss_type,
                device=self.device,
                lagrangian_config=lagrangian_config,
                log_normalized_violation=self.log_normalized_violation,
            )

        if self.global_rank == 0:
            print(f"Loss Managers initialized with loss_type='{self.loss_type}'")

    def _default_wandb_run_name(self):
        case_tag = f"{len(self.case_names)}cases"
        return f"acopf-ddp-{case_tag}-{self.model_type}-{self.loss_type}"

    def _should_print_epoch(self):
        return self.global_rank == 0 and not self.wandb_enabled

    def _checkpoint_tag(self):
        if len(self.case_names) == 1:
            return self.case_names[0]
        return f"multi{len(self.case_names)}cases"

    def _checkpoint_payload(self):
        payload = super()._checkpoint_payload()
        payload["case_names"] = list(self.case_names)
        return payload

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        total_task_loss = 0.0
        num_batches = 0
        metric_sums, metric_counts = self._init_metric_trackers(self.train_metric_names)
        tracker = self.throughput_tracker

        for case_idx in self.train_case_indices:
            loader = self.train_loaders[case_idx]
            sampler = self.train_samplers.get(case_idx)
            if sampler is not None:
                sampler.set_epoch(epoch)

            total_steps = len(loader)
            if total_steps == 0:
                continue

            if self.global_rank == 0 and not self.wandb_enabled:
                case_name = self.case_names[case_idx]
                pbar = tqdm(loader, desc=f"Epoch {epoch} {case_name}")
            else:
                pbar = loader

            self.optimizer.zero_grad()
            accum_batches = 0
            step_start_time = None
            step_samples = 0

            for batch_idx, batch in enumerate(pbar):
                is_step_start = accum_batches == 0
                if is_step_start and tracker:
                    tracker.maybe_start_measurement()
                    if tracker.measure_active():
                        tracker.accelerator_synchronize()
                        step_start_time = time.perf_counter()
                        step_samples = 0

                batch = batch.to(self.device)
                batch_samples = self._update_global_samples(batch)

                if tracker and tracker.measure_active():
                    step_samples += tracker.get_batch_samples(batch)

                predictions = self.forward(batch)
                collect_constraints = self.loss_managers[case_idx].lagrangian is not None
                loss, loss_info = self.loss_managers[case_idx].compute_loss(
                    predictions,
                    batch,
                    return_info=True,
                    collect_constraints=collect_constraints,
                )

                global_batch_samples = batch_samples
                synced_violation = loss_info.get("constraint_violation")
                synced_constraints = loss_info.get("constraints")
                if self.loss_managers[case_idx].lagrangian is not None:
                    global_batch_samples = self._sync_batch_samples(batch_samples)
                    synced_violation, synced_constraints = self._sync_lagrangian_inputs(
                        synced_violation,
                        synced_constraints,
                        batch_samples=batch_samples,
                        total_batch_samples=global_batch_samples,
                    )

                self.loss_managers[case_idx].update_lagrangian(
                    constraint_violation=synced_violation,
                    constraints=synced_constraints,
                    is_training=self.model.training,
                    sample_count=global_batch_samples,
                )

                loss_value = loss.item()
                self._add_metric(metric_sums, metric_counts, "train/loss/total", loss_value)
                if self.log_every_n_samples and self.log_every_n_samples > 0:
                    self._log_wandb_step(loss_value, loss_info)
                loss = loss / self.accumulate_grad_batches
                loss.backward()

                should_step = ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (
                    (batch_idx + 1) == total_steps
                )

                if should_step:
                    if tracker and tracker.measure_active():
                        tracker.accelerator_synchronize()
                    self._clip_gradients()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    if not self.log_every_n_samples or self.log_every_n_samples <= 0:
                        self._log_wandb_step(loss_value, loss_info)
                    if tracker:
                        step_metrics = None
                        if tracker.measure_active() and step_start_time is not None:
                            tracker.accelerator_synchronize()
                            step_time = time.perf_counter() - step_start_time
                            total_samples = step_samples * self.world_size
                            samples_per_sec = total_samples / step_time if step_time > 0 else 0.0
                            step_metrics = {"throughput/samples_per_sec": samples_per_sec}
                        tracker.on_step_end(step_metrics)
                    accum_batches = 0
                else:
                    accum_batches += 1

                total_loss += loss_value
                if "objective" in loss_info:
                    objective_value = self._as_float(loss_info["objective"])
                    if objective_value is not None:
                        total_task_loss += objective_value
                        self._add_metric(metric_sums, metric_counts, "train/loss/objective", objective_value)
                for info_key, metric_name in self.train_metric_map:
                    if info_key in loss_info:
                        self._add_metric(metric_sums, metric_counts, metric_name, loss_info[info_key])
                num_batches += 1

                if self.global_rank == 0 and not self.wandb_enabled:
                    if self.log_every_n_steps and self.log_every_n_steps > 0:
                        if self.global_step % self.log_every_n_steps == 0:
                            pbar.set_postfix({"loss": loss_value})
                    else:
                        pbar.set_postfix({"loss": loss_value})

                if self._maybe_run_validation() and self.stop_training:
                    break
                self._maybe_save_periodic_checkpoint()
                if self._maybe_stop_by_samples():
                    break

            if self.stop_training:
                break

        if num_batches == 0:
            return 0.0, 0.0, {}

        avg_loss = total_loss / num_batches
        avg_task_loss = total_task_loss / num_batches

        loss_tensor = torch.tensor([avg_loss, avg_task_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

        metric_sums, metric_counts = self._reduce_metrics(metric_sums, metric_counts)
        metric_avgs = self._compute_metric_avgs(metric_sums, metric_counts)

        for manager in self.loss_managers.values():
            manager.step_epoch()

        return loss_tensor[0].item(), loss_tensor[1].item(), metric_avgs

    def validate(self):
        if not self.val_case_indices:
            return None, None, None

        self.model.eval()

        total_loss = 0.0
        total_task_loss = 0.0
        num_batches = 0
        metric_sums, metric_counts = self._init_metric_trackers(self.val_metric_names)

        with torch.no_grad():
            for case_idx in self.val_case_indices:
                loader = self.val_loaders[case_idx]
                if loader is None:
                    continue

                for batch in loader:
                    batch = batch.to(self.device)

                    predictions = self.forward(batch)
                    loss, loss_info = self.loss_managers[case_idx].compute_loss(
                        predictions,
                        batch,
                        return_info=True,
                    )

                    loss_value = loss.item()
                    total_loss += loss_value
                    self._add_metric(metric_sums, metric_counts, "val/loss/total", loss_value)
                    if "objective" in loss_info:
                        objective_value = self._as_float(loss_info["objective"])
                    if objective_value is not None:
                        total_task_loss += objective_value
                        self._add_metric(metric_sums, metric_counts, "val/loss/objective", objective_value)
                    for info_key, metric_name in self.val_metric_map:
                        if info_key in loss_info:
                            self._add_metric(metric_sums, metric_counts, metric_name, loss_info[info_key])
                    num_batches += 1

        if num_batches == 0:
            return None, None, None

        avg_loss = total_loss / num_batches
        avg_task_loss = total_task_loss / num_batches

        loss_tensor = torch.tensor([avg_loss, avg_task_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

        metric_sums, metric_counts = self._reduce_metrics(metric_sums, metric_counts)
        metric_avgs = self._compute_metric_avgs(metric_sums, metric_counts)
        self._add_val_score(metric_avgs)

        return loss_tensor[0].item(), loss_tensor[1].item(), metric_avgs
