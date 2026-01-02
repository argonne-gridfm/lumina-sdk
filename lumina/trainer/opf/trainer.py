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

from lumina.dataset.opf.opf_dataset import OPFDataset, OPFHomogeneousDataset, OPFMultiDataset
from lumina.dataset.opf.opf_on_disk_dataset import OPFOnDiskDataset, OPFOnDiskHomogeneousDataset
from lumina.dataset.opf.opf_sharded_dataset import (
    OPFShardedIterableDataset,
    build_shard_infos,
    filter_shards_by_group,
    load_shard_manifest,
    resolve_split_shards,
    split_shards_by_ratio,
)
from lumina.dataset.opf.case_id import CaseTaggedDataset, CaseTaggedIterableDataset
from lumina.dataset.opf.staging import (
    get_on_disk_db_path,
    get_on_disk_lock_path,
    get_sharded_lock_path,
    get_sharded_manifest_path,
    file_lock,
    resolve_stage_root,
    stage_on_disk_group,
    stage_sharded_case,
)
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
        "train/perf/constraint_eval_ms",
    )
    VAL_METRIC_NAMES = (
        "val/loss/total",
        "val/loss/objective",
        "val/feas/total_violation",
        "val/feas/total_violation_norm",
        "val/feas/p_balance_rmse_pu",
        "val/feas/q_balance_rmse_pu",
        "val/feas/line_limit_rmse_pu",
        "val/perf/constraint_eval_ms",
        "val/perf/eval_batches",
        "val/perf/data_ms",
        "val/perf/forward_ms",
        "val/perf/loss_ms",
        "val/perf/total_ms",
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
        ("constraint_eval_ms", "train/perf/constraint_eval_ms"),
    )
    VAL_METRIC_MAP = (
        ("raw_constraint_violation", "val/feas/total_violation"),
        ("raw_constraint_violation_norm", "val/feas/total_violation_norm"),
        ("p_balance_rmse", "val/feas/p_balance_rmse_pu"),
        ("q_balance_rmse", "val/feas/q_balance_rmse_pu"),
        ("line_limit_rmse", "val/feas/line_limit_rmse_pu"),
        ("constraint_eval_ms", "val/perf/constraint_eval_ms"),
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
        ("train/perf/constraint_eval_ms",),
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
        (
            "val/perf/constraint_eval_ms",
            "val/perf/eval_batches",
            "val/perf/data_ms",
            "val/perf/forward_ms",
            "val/perf/loss_ms",
            "val/perf/total_ms",
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
        data_config = self.config.get("data", {})
        if "use_precomputed_homo" in data_config:
            use_precomputed = bool(data_config.get("use_precomputed_homo"))
        else:
            use_precomputed = True
        self.use_precomputed_homo = use_precomputed and self.model_type not in HETERO_MODEL_TYPES
        self.dataset_backend = str(data_config.get("backend", "in_memory")).lower()
        self.on_disk_backend = str(data_config.get("on_disk_backend", "sqlite")).lower()
        self.on_disk_write_batch_size = int(data_config.get("on_disk_write_batch_size", 128))
        self.on_disk_sqlite_timeout_sec = float(data_config.get("on_disk_sqlite_timeout_sec", 600.0))
        sqlite_busy_timeout = data_config.get("on_disk_sqlite_busy_timeout_ms")
        self.on_disk_sqlite_busy_timeout_ms = (
            int(sqlite_busy_timeout) if sqlite_busy_timeout is not None else None
        )
        self.on_disk_sqlite_journal_mode = data_config.get("on_disk_sqlite_journal_mode", "WAL")
        self.on_disk_sqlite_synchronous = data_config.get("on_disk_sqlite_synchronous", "NORMAL")
        self.topological_perturbations = bool(data_config.get("topological_perturbations", False))
        self.data_staging = data_config.get("staging", {}) if isinstance(data_config.get("staging"), dict) else {}
        self.data_staging_lock_timeout = int(self.data_staging.get("lock_timeout_sec", 7200))
        self.on_disk_homo_suffix = str(data_config.get("on_disk_homo_suffix", "homo"))
        self.on_disk_homo_prune = bool(data_config.get("on_disk_homo_prune", True))
        self.on_disk_homo_storage_dtype = data_config.get("on_disk_homo_storage_dtype", "float16")
        self.on_disk_homo_restore_fp32 = bool(data_config.get("on_disk_homo_restore_fp32", True))
        self.on_disk_homo_attach_full_edge_attr = bool(
            data_config.get("on_disk_homo_attach_full_edge_attr", False)
        )
        self.on_disk_homo_sanitize_targets = bool(data_config.get("on_disk_homo_sanitize_targets", True))
        self.on_disk_homo_log_bad_targets = bool(data_config.get("on_disk_homo_log_bad_targets", True))
        self.on_disk_homo_max_bad_target_logs = int(data_config.get("on_disk_homo_max_bad_target_logs", 1))
        self.sharded_root = data_config.get("sharded_root")
        self.sharded_manifest_name = str(data_config.get("sharded_manifest_name", "manifest.json"))
        self.sharded_suffix = data_config.get("sharded_suffix")
        self.sharded_homo_suffix = data_config.get("sharded_homo_suffix", self.on_disk_homo_suffix)
        split_seed = data_config.get("sharded_split_seed", self.config.get("split_seed", 42))
        self.sharded_split_seed = int(split_seed)
        self.homo_dataset_kwargs = {}
        if isinstance(data_config.get("homo_dataset_kwargs"), dict):
            self.homo_dataset_kwargs.update(data_config["homo_dataset_kwargs"])
        default_homo_kwargs = {
            "processed_suffix": self.on_disk_homo_suffix,
            "attach_full_edge_attr": self.on_disk_homo_attach_full_edge_attr,
            "sanitize_targets": self.on_disk_homo_sanitize_targets,
            "log_bad_targets": self.on_disk_homo_log_bad_targets,
            "max_bad_target_logs": self.on_disk_homo_max_bad_target_logs,
        }
        for key, value in default_homo_kwargs.items():
            self.homo_dataset_kwargs.setdefault(key, value)

        training_config = self.config["training"]
        self.max_epochs = training_config["max_epochs"]
        self.patience = training_config["patience"]
        self.grad_clip_val = training_config.get("gradient_clip_val")
        self.grad_clip_algo = training_config.get("gradient_clip_algorithm", "norm")
        self.accumulate_grad_batches = max(1, int(training_config.get("accumulate_grad_batches", 1)))
        case_mix_every = training_config.get("case_mix_every_n_steps", 0)
        try:
            case_mix_every = int(case_mix_every)
        except (TypeError, ValueError):
            case_mix_every = 0
        self.case_mix_every_n_steps = max(0, case_mix_every)
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
        violation_eval_p = training_config.get("violation_eval_p", 1.0)
        self.violation_eval_p = float(1.0 if violation_eval_p is None else violation_eval_p)
        if self.violation_eval_p < 0.0:
            self.violation_eval_p = 0.0
        elif self.violation_eval_p > 1.0:
            self.violation_eval_p = 1.0
        self.validation_timing = bool(training_config.get("validation_timing", False))
        validation_timing_every = training_config.get("validation_timing_every_n_batches", 1)
        try:
            validation_timing_every = int(validation_timing_every)
        except (TypeError, ValueError):
            validation_timing_every = 1
        self.validation_timing_every_n_batches = max(1, validation_timing_every)
        validation_timing_max = training_config.get("validation_timing_max_batches", 0)
        try:
            validation_timing_max = int(validation_timing_max)
        except (TypeError, ValueError):
            validation_timing_max = 0
        self.validation_timing_max_batches = max(0, validation_timing_max)
        min_eval_batches = training_config.get("violation_eval_min_batches", 1)
        self.violation_eval_min_batches = max(0, int(min_eval_batches or 0))
        violation_eval_seed = training_config.get("violation_eval_seed")
        self.violation_eval_seed = None if violation_eval_seed is None else int(violation_eval_seed)

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

    def _infer_output_dim(self, sample_data):
        if hasattr(sample_data, "node_types"):
            y = getattr(sample_data["bus"], "y", None)
        else:
            y = getattr(sample_data, "y", None)
        if y is None:
            raise ValueError("Unable to infer per-node output size from dataset sample.")
        if y.ndim <= 1:
            return 1
        return int(y.shape[-1])

    def _use_on_disk_backend(self):
        return self.dataset_backend == "on_disk"

    def _use_sharded_backend(self):
        return self.dataset_backend == "sharded"

    def _select_dataset_cls(self):
        if self._use_on_disk_backend():
            if self.model_type in HETERO_MODEL_TYPES:
                return OPFOnDiskDataset
            if self.use_precomputed_homo:
                return OPFOnDiskHomogeneousDataset
            return OPFOnDiskDataset
        return OPFHomogeneousDataset if self.use_precomputed_homo else OPFDataset

    def _resolve_sharded_root(self):
        return self.sharded_root or self.config["root"]

    def _sharded_processed_suffix(self):
        if self.use_precomputed_homo:
            return self.sharded_homo_suffix
        return self.sharded_suffix

    def _stage_on_disk(self, case_name, group_ids, dataset_cls, build_kwargs, processed_suffix=None):
        if not self._use_on_disk_backend():
            return self.config["root"]

        if not self.data_staging.get("enabled", False):
            return self.config["root"]

        stage_root = resolve_stage_root(self.data_staging)
        if not stage_root:
            if self.global_rank == 0:
                print("Warning: staging enabled but no stage root resolved; using shared root.")
            return self.config["root"]

        source_root = self.config["root"]
        if os.path.abspath(stage_root) == os.path.abspath(source_root):
            return source_root

        for group_id in group_ids:
            src_path = get_on_disk_db_path(
                source_root,
                case_name,
                group_id,
                self.on_disk_backend,
                self.topological_perturbations,
                processed_suffix,
            )
            lock_path = get_on_disk_lock_path(
                source_root,
                case_name,
                group_id,
                self.on_disk_backend,
                self.topological_perturbations,
                processed_suffix,
            )
            if self.global_rank == 0:
                with file_lock(lock_path, timeout_sec=self.data_staging_lock_timeout):
                    if not os.path.exists(src_path):
                        print(f"On-disk dataset missing at {src_path}; building on shared root.")
                        dataset = dataset_cls(group_id=group_id, **build_kwargs, log=True)
                        dataset.close()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if self.local_rank == 0:
                with file_lock(lock_path, timeout_sec=self.data_staging_lock_timeout):
                    stage_on_disk_group(
                        source_root=source_root,
                        stage_root=stage_root,
                        case_name=case_name,
                        group_id=group_id,
                        backend=self.on_disk_backend,
                        topological_perturbations=self.topological_perturbations,
                        processed_suffix=processed_suffix,
                        log=self.global_rank == 0,
                    )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        return stage_root

    def _stage_sharded(self, case_name, processed_suffix=None):
        if not self._use_sharded_backend():
            return self._resolve_sharded_root()

        if not self.data_staging.get("enabled", False):
            return self._resolve_sharded_root()

        stage_root = resolve_stage_root(self.data_staging)
        if not stage_root:
            if self.global_rank == 0:
                print("Warning: staging enabled but no stage root resolved; using shared root.")
            return self._resolve_sharded_root()

        source_root = self._resolve_sharded_root()
        if os.path.abspath(stage_root) == os.path.abspath(source_root):
            return source_root

        manifest_path = get_sharded_manifest_path(
            source_root,
            case_name,
            self.topological_perturbations,
            processed_suffix,
            self.sharded_manifest_name,
        )
        lock_path = get_sharded_lock_path(
            source_root,
            case_name,
            self.topological_perturbations,
            processed_suffix,
            self.sharded_manifest_name,
        )

        if self.global_rank == 0:
            with file_lock(lock_path, timeout_sec=self.data_staging_lock_timeout):
                if not os.path.exists(manifest_path):
                    raise FileNotFoundError(
                        f"Sharded manifest missing at {manifest_path}. "
                        "Run scripts/opf_build_shards.py first."
                    )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if self.local_rank == 0:
            with file_lock(lock_path, timeout_sec=self.data_staging_lock_timeout):
                stage_sharded_case(
                    source_root=source_root,
                    stage_root=stage_root,
                    case_name=case_name,
                    topological_perturbations=self.topological_perturbations,
                    processed_suffix=processed_suffix,
                    manifest_name=self.sharded_manifest_name,
                    log=self.global_rank == 0,
                )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        return stage_root

    def _is_on_disk_dataset_cls(self, dataset_cls) -> bool:
        try:
            return issubclass(dataset_cls, OPFOnDiskDataset)
        except TypeError:
            return False

    def _make_dataset_kwargs(self, dataset_cls, root, case_name):
        dataset_kwargs = dict(
            root=root,
            case_name=case_name,
            topological_perturbations=self.topological_perturbations,
            local_raw_folder=self.config.get("local_raw_folder"),
            force_reload=False,
        )

        if self._is_on_disk_dataset_cls(dataset_cls):
            dataset_kwargs.update(
                {
                    "backend": self.on_disk_backend,
                    "topological_perturbations": self.topological_perturbations,
                    "write_batch_size": self.on_disk_write_batch_size,
                    "sqlite_timeout_sec": self.on_disk_sqlite_timeout_sec,
                    "sqlite_busy_timeout_ms": self.on_disk_sqlite_busy_timeout_ms,
                    "sqlite_journal_mode": self.on_disk_sqlite_journal_mode,
                    "sqlite_synchronous": self.on_disk_sqlite_synchronous,
                }
            )

        if dataset_cls is OPFOnDiskHomogeneousDataset:
            dataset_kwargs.update(
                {
                    "processed_suffix": self.on_disk_homo_suffix,
                    "prune_homo": self.on_disk_homo_prune,
                    "storage_dtype": self.on_disk_homo_storage_dtype,
                    "restore_fp32": self.on_disk_homo_restore_fp32,
                    "attach_full_edge_attr": self.on_disk_homo_attach_full_edge_attr,
                    "sanitize_targets": self.on_disk_homo_sanitize_targets,
                    "log_bad_targets": self.on_disk_homo_log_bad_targets,
                    "max_bad_target_logs": self.on_disk_homo_max_bad_target_logs,
                }
            )
        elif dataset_cls is OPFHomogeneousDataset and self.homo_dataset_kwargs:
            dataset_kwargs.update(self.homo_dataset_kwargs)

        return dataset_kwargs

    def _log_dataset_choice(
        self,
        case_name,
        dataset_cls,
        dataset_root,
        processed_suffix=None,
        manifest_path=None,
    ):
        if self.global_rank != 0:
            return
        dataset_name = dataset_cls.__name__ if dataset_cls is not None else "OPFShardedIterableDataset"
        parts = [
            f"backend={self.dataset_backend}",
            f"dataset_cls={dataset_name}",
            f"root={dataset_root}",
        ]
        if self.dataset_backend == "on_disk":
            parts.append(f"on_disk_backend={self.on_disk_backend}")
        if processed_suffix:
            parts.append(f"processed_suffix={processed_suffix}")
        if manifest_path:
            parts.append(f"manifest={manifest_path}")
        print(f"Dataset config ({case_name}): " + ", ".join(parts))

    def _load_sharded_splits(self, case_name, group_ids):
        processed_suffix = self._sharded_processed_suffix()
        dataset_root = self._stage_sharded(case_name, processed_suffix)
        manifest_path = get_sharded_manifest_path(
            dataset_root,
            case_name,
            self.topological_perturbations,
            processed_suffix,
            self.sharded_manifest_name,
        )
        self._log_dataset_choice(
            case_name,
            OPFShardedIterableDataset,
            dataset_root,
            processed_suffix=processed_suffix,
            manifest_path=manifest_path,
        )
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Sharded manifest not found at {manifest_path}. "
                "Run scripts/opf_build_shards.py first."
            )
        manifest = load_shard_manifest(manifest_path)
        all_shards = build_shard_infos(manifest)

        splits = {}
        if "splits" in manifest:
            for split in ("train", "val", "test"):
                try:
                    split_shards = resolve_split_shards(manifest, all_shards, split)
                except KeyError:
                    split_shards = []
                split_shards = filter_shards_by_group(split_shards, group_ids)
                splits[split] = split_shards
        else:
            filtered = filter_shards_by_group(all_shards, group_ids)
            splits = split_shards_by_ratio(
                filtered,
                self.config["train_split"],
                self.config["val_split"],
                seed=self.sharded_split_seed,
                shuffle=True,
            )

        if not splits.get("train"):
            raise ValueError("Sharded dataset has no training shards after filtering.")
        splits.setdefault("val", [])
        splits.setdefault("test", [])
        return splits

    def _loader_kwargs(self, loader_config):
        num_workers = int(loader_config.get("num_workers", 0))
        kwargs = {
            "batch_size": loader_config["batch_size"],
            "num_workers": num_workers,
            "pin_memory": bool(loader_config.get("pin_memory", True)),
        }
        if num_workers > 0:
            prefetch_factor = loader_config.get("prefetch_factor")
            if prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(prefetch_factor)
            persistent_workers = loader_config.get("persistent_workers")
            if persistent_workers is not None:
                kwargs["persistent_workers"] = bool(persistent_workers)
        return kwargs

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
        if hasattr(sample_data, "x") and hasattr(sample_data, "edge_index"):
            return sample_data
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

    def _sync_for_timing(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif hasattr(torch, "accelerator") and hasattr(torch.accelerator, "synchronize"):
            torch.accelerator.synchronize()

    def _should_time_validation_batch(self, batch_idx, timed_batches):
        if not self.validation_timing:
            return False
        if self.validation_timing_every_n_batches > 1:
            if batch_idx % self.validation_timing_every_n_batches != 0:
                return False
        if self.validation_timing_max_batches and timed_batches >= self.validation_timing_max_batches:
            return False
        return True

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

    def _violation_eval_disabled(self):
        return self.violation_eval_p <= 0.0

    def _make_violation_eval_rng(self, case_idx=0):
        seed = self.violation_eval_seed
        if seed is None:
            return np.random.RandomState()
        offset = int(self.global_samples) + int(self.current_epoch) * 1000 + int(case_idx) * 100000
        seed = (int(seed) + offset) % (2**32)
        return np.random.RandomState(seed)

    def _should_eval_violations(self, rng, batch_idx, total_batches, eval_batches, min_eval_batches):
        if total_batches is not None and total_batches > 0:
            remaining_batches = total_batches - batch_idx
            remaining_needed = min_eval_batches - eval_batches
            if remaining_needed > 0 and remaining_batches <= remaining_needed:
                return True
        return rng.random_sample() < self.violation_eval_p

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
        if self._use_sharded_backend():
            self.sharded_splits = self._load_sharded_splits(self.case_name, self.group_ids)
            if self.global_rank == 0:
                counts = {
                    split: sum(shard.num_samples for shard in shards)
                    for split, shards in self.sharded_splits.items()
                }
                print(
                    "Sharded dataset loaded: "
                    f"train={counts.get('train', 0)}, "
                    f"val={counts.get('val', 0)}, "
                    f"test={counts.get('test', 0)} samples"
                )
            return
        dataset_cls = self._select_dataset_cls()
        build_kwargs = self._make_dataset_kwargs(dataset_cls, self.config["root"], self.case_name)
        processed_suffix = self.on_disk_homo_suffix if dataset_cls is OPFOnDiskHomogeneousDataset else None
        dataset_root = self._stage_on_disk(
            self.case_name,
            self.group_ids,
            dataset_cls,
            build_kwargs,
            processed_suffix,
        )
        self._log_dataset_choice(self.case_name, dataset_cls, dataset_root, processed_suffix=processed_suffix)
        dataset_kwargs = dict(build_kwargs)
        dataset_kwargs["root"] = dataset_root

        def build_dataset():
            if len(self.group_ids) == 1:
                return dataset_cls(group_id=self.group_ids[0], **dataset_kwargs)
            return OPFMultiDataset.from_case_groups(
                group_ids=self.group_ids,
                dataset_cls=dataset_cls,
                **dataset_kwargs,
            )

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
        if self._use_sharded_backend():
            loader_config = self.config["loader"]
            if self.model_type not in HETERO_MODEL_TYPES and not self.use_precomputed_homo:
                raise ValueError(
                    "Sharded backend requires precomputed homogeneous shards when using homo models. "
                    "Set use_precomputed_homo=true or switch backend."
                )
            case_id = 0
            self.train_dataset = CaseTaggedIterableDataset(
                OPFShardedIterableDataset(
                    self.sharded_splits["train"],
                    shuffle_shards=loader_config["shuffle"],
                    seed=self.sharded_split_seed,
                ),
                case_id,
            )
            self.val_dataset = CaseTaggedIterableDataset(
                OPFShardedIterableDataset(
                    self.sharded_splits.get("val", []),
                    shuffle_shards=False,
                    seed=self.sharded_split_seed,
                ),
                case_id,
            )
            self.test_dataset = CaseTaggedIterableDataset(
                OPFShardedIterableDataset(
                    self.sharded_splits.get("test", []),
                    shuffle_shards=False,
                    seed=self.sharded_split_seed,
                ),
                case_id,
            )
            self.train_sampler = None
            self.val_sampler = None
            self.train_loader = DataLoader(self.train_dataset, **self._loader_kwargs(loader_config))
            self.val_loader = DataLoader(self.val_dataset, **self._loader_kwargs(loader_config))
            self.test_loader = DataLoader(self.test_dataset, **self._loader_kwargs(loader_config))
            self.dataset = self.train_dataset
            return

        n_samples = len(self.dataset)
        n_train = int(n_samples * self.config["train_split"])
        n_val = int(n_samples * self.config["val_split"])

        train_dataset = torch.utils.data.Subset(self.dataset, range(n_train))
        val_dataset = torch.utils.data.Subset(self.dataset, range(n_train, n_train + n_val))
        test_dataset = torch.utils.data.Subset(self.dataset, range(n_train + n_val, n_samples))

        if self.model_type not in HETERO_MODEL_TYPES and not self.use_precomputed_homo:
            train_dataset = HomoOPFDataset(train_dataset)
            val_dataset = HomoOPFDataset(val_dataset)
            test_dataset = HomoOPFDataset(test_dataset)

        case_id = 0
        train_dataset = CaseTaggedDataset(train_dataset, case_id)
        val_dataset = CaseTaggedDataset(val_dataset, case_id)
        test_dataset = CaseTaggedDataset(test_dataset, case_id)

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
            sampler=self.train_sampler,
            **self._loader_kwargs(loader_config),
        )

        self.val_loader = DataLoader(
            val_dataset,
            sampler=self.val_sampler,
            **self._loader_kwargs(loader_config),
        )

        self.test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **self._loader_kwargs(loader_config),
        )

    def _create_model(self):
        if self._use_sharded_backend():
            sample_data = self.train_dataset.peek()
            metadata = self.train_dataset.metadata() if self.model_type in HETERO_MODEL_TYPES else None
        else:
            sample_data = self.dataset[0]
            metadata = self.dataset.metadata() if self.model_type in HETERO_MODEL_TYPES else None
        per_node_output_size = self._infer_output_dim(sample_data)
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
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        elif hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(epoch)

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

        if self.violation_eval_p <= 0.0:
            return None, None, None

        total_loss = 0.0
        total_task_loss = 0.0
        num_batches = 0
        metric_sums, metric_counts = self._init_metric_trackers(self.val_metric_names)
        timed_batches = 0
        rng = self._make_violation_eval_rng()
        try:
            total_batches = len(self.val_loader)
        except Exception:
            total_batches = None
        min_eval_batches = self.violation_eval_min_batches
        if total_batches is not None and total_batches > 0:
            min_eval_batches = min(min_eval_batches, total_batches)
        eval_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                do_eval = self._should_eval_violations(
                    rng,
                    batch_idx,
                    total_batches,
                    eval_batches,
                    min_eval_batches,
                )
                if not do_eval:
                    continue
                eval_batches += 1

                do_timing = self._should_time_validation_batch(batch_idx, timed_batches)
                if do_timing:
                    self._sync_for_timing()
                    timing_start = time.perf_counter()
                batch = batch.to(self.device)
                if do_timing:
                    self._sync_for_timing()
                    timing_after_data = time.perf_counter()

                predictions = self.forward(batch)
                if do_timing:
                    self._sync_for_timing()
                    timing_after_forward = time.perf_counter()
                if self.loss_manager.lagrangian is None:
                    loss, loss_info = self.loss_manager.compute_loss(
                        predictions,
                        batch,
                        return_info=True,
                        collect_constraints=True,
                    )
                else:
                    loss, loss_info = self.loss_manager.compute_loss(
                        predictions,
                        batch,
                        return_info=True,
                    )
                if do_timing:
                    self._sync_for_timing()
                    timing_after_loss = time.perf_counter()
                    self._add_metric(
                        metric_sums,
                        metric_counts,
                        "val/perf/data_ms",
                        (timing_after_data - timing_start) * 1000.0,
                    )
                    self._add_metric(
                        metric_sums,
                        metric_counts,
                        "val/perf/forward_ms",
                        (timing_after_forward - timing_after_data) * 1000.0,
                    )
                    self._add_metric(
                        metric_sums,
                        metric_counts,
                        "val/perf/loss_ms",
                        (timing_after_loss - timing_after_forward) * 1000.0,
                    )
                    self._add_metric(
                        metric_sums,
                        metric_counts,
                        "val/perf/total_ms",
                        (timing_after_loss - timing_start) * 1000.0,
                    )
                    timed_batches += 1

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

        totals = torch.tensor(
            [total_loss, total_task_loss, float(num_batches)],
            device=self.device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, total_task_loss, total_batches = totals.tolist()
        if total_batches == 0:
            return None, None, None

        avg_loss = total_loss / total_batches
        avg_task_loss = total_task_loss / total_batches

        metric_sums, metric_counts = self._reduce_metrics(metric_sums, metric_counts)
        metric_avgs = self._compute_metric_avgs(metric_sums, metric_counts)
        metric_avgs["val/perf/eval_batches"] = float(total_batches)
        self._add_val_score(metric_avgs)

        return avg_loss, avg_task_loss, metric_avgs


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
        dataset_cls = self._select_dataset_cls()
        build_kwargs = self._make_dataset_kwargs(dataset_cls, self.config["root"], case_name)
        processed_suffix = self.on_disk_homo_suffix if dataset_cls is OPFOnDiskHomogeneousDataset else None
        dataset_root = self._stage_on_disk(
            case_name,
            self.group_ids,
            dataset_cls,
            build_kwargs,
            processed_suffix,
        )
        self._log_dataset_choice(case_name, dataset_cls, dataset_root, processed_suffix=processed_suffix)
        dataset_kwargs = dict(build_kwargs)
        dataset_kwargs["root"] = dataset_root

        def build_dataset():
            if len(self.group_ids) == 1:
                return dataset_cls(group_id=self.group_ids[0], **dataset_kwargs)
            return OPFMultiDataset.from_case_groups(
                group_ids=self.group_ids,
                dataset_cls=dataset_cls,
                **dataset_kwargs,
            )
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
        if self._use_sharded_backend():
            self.case_sharded_splits = {}
            reference_metadata = None
            reference_out_dim = None

            for case_idx, case_name in enumerate(self.case_names):
                splits = self._load_sharded_splits(case_name, self.group_ids)
                self.case_sharded_splits[case_idx] = splits

                sample_shards = splits["train"] or splits.get("val", []) or splits.get("test", [])
                if not sample_shards:
                    raise ValueError(f"Sharded dataset for case {case_name} is empty.")

                sample_dataset = OPFShardedIterableDataset(sample_shards, shuffle_shards=False)
                sample = sample_dataset.peek()
                out_dim = self._infer_output_dim(sample)

                if self.model_type in HETERO_MODEL_TYPES:
                    metadata = sample_dataset.metadata()
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
                else:
                    if reference_out_dim is None:
                        reference_out_dim = out_dim
                    elif out_dim != reference_out_dim:
                        raise ValueError(
                            f"Output dimension mismatch for case {case_name} "
                            f"(expected {reference_out_dim}, found {out_dim})."
                        )

                if self.global_rank == 0:
                    counts = {split: sum(shard.num_samples for shard in shards) for split, shards in splits.items()}
                    print(
                        f"Sharded dataset loaded for {case_name}: "
                        f"train={counts.get('train', 0)}, "
                        f"val={counts.get('val', 0)}, "
                        f"test={counts.get('test', 0)} samples"
                    )

            self.reference_metadata = reference_metadata
            self.reference_output_dim = reference_out_dim
            return

        self.case_datasets = []
        reference_metadata = None
        reference_out_dim = None

        for case_name in self.case_names:
            dataset = self._load_dataset(case_name)
            if len(dataset) == 0:
                raise ValueError(f"Dataset for case {case_name} is empty.")

            if self.global_rank == 0:
                print(f"Dataset loaded for {case_name}: {len(dataset)} samples")

            sample = dataset[0]
            out_dim = self._infer_output_dim(sample)

            if self.model_type in HETERO_MODEL_TYPES:
                metadata = dataset.metadata()
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
            else:
                if reference_out_dim is None:
                    reference_out_dim = out_dim
                elif out_dim != reference_out_dim:
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

        if self._use_sharded_backend():
            if self.model_type not in HETERO_MODEL_TYPES and not self.use_precomputed_homo:
                raise ValueError(
                    "Sharded backend requires precomputed homogeneous shards when using homo models. "
                    "Set use_precomputed_homo=true or switch backend."
                )
            for case_idx, splits in self.case_sharded_splits.items():
                train_shards = splits.get("train", [])
                val_shards = splits.get("val", [])
                test_shards = splits.get("test", [])
                if not train_shards:
                    continue
                train_dataset = CaseTaggedIterableDataset(
                    OPFShardedIterableDataset(
                        train_shards,
                        shuffle_shards=loader_config["shuffle"],
                        seed=self.sharded_split_seed + case_idx,
                    ),
                    case_idx,
                )
                self.train_samplers[case_idx] = None
                self.train_loaders[case_idx] = DataLoader(
                    train_dataset,
                    **self._loader_kwargs(loader_config),
                )
                self.train_case_indices.append(case_idx)

                if val_shards:
                    val_dataset = CaseTaggedIterableDataset(
                        OPFShardedIterableDataset(
                            val_shards,
                            shuffle_shards=False,
                            seed=self.sharded_split_seed + case_idx,
                        ),
                        case_idx,
                    )
                    self.val_samplers[case_idx] = None
                    self.val_loaders[case_idx] = DataLoader(
                        val_dataset,
                        **self._loader_kwargs(loader_config),
                    )
                    self.val_case_indices.append(case_idx)

                if test_shards:
                    test_dataset = CaseTaggedIterableDataset(
                        OPFShardedIterableDataset(
                            test_shards,
                            shuffle_shards=False,
                            seed=self.sharded_split_seed + case_idx,
                        ),
                        case_idx,
                    )
                    self.test_loaders[case_idx] = DataLoader(
                        test_dataset,
                        **self._loader_kwargs(loader_config),
                    )
                    self.test_case_indices.append(case_idx)
            return

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

            if self.model_type not in HETERO_MODEL_TYPES and not self.use_precomputed_homo:
                train_dataset = HomoOPFDataset(train_dataset)
                if val_len > 0:
                    val_dataset = HomoOPFDataset(val_dataset)
                if test_len > 0:
                    test_dataset = HomoOPFDataset(test_dataset)

            train_dataset = CaseTaggedDataset(train_dataset, case_idx)
            if val_len > 0:
                val_dataset = CaseTaggedDataset(val_dataset, case_idx)
            if test_len > 0:
                test_dataset = CaseTaggedDataset(test_dataset, case_idx)

            self.train_samplers[case_idx] = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=loader_config["shuffle"],
            )
            self.train_loaders[case_idx] = DataLoader(
                train_dataset,
                sampler=self.train_samplers[case_idx],
                **self._loader_kwargs(loader_config),
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
                    sampler=self.val_samplers[case_idx],
                    **self._loader_kwargs(loader_config),
                )
                self.val_case_indices.append(case_idx)

            if test_len > 0:
                self.test_loaders[case_idx] = DataLoader(
                    test_dataset,
                    shuffle=False,
                    **self._loader_kwargs(loader_config),
                )
                self.test_case_indices.append(case_idx)

    def _create_model(self):
        if self._use_sharded_backend():
            first_case = sorted(self.case_sharded_splits.keys())[0]
            splits = self.case_sharded_splits[first_case]
            sample_shards = splits.get("train", []) or splits.get("val", []) or splits.get("test", [])
            sample_dataset = OPFShardedIterableDataset(sample_shards, shuffle_shards=False)
            sample_data = sample_dataset.peek()
        else:
            sample_data = self.case_datasets[0][0]
        per_node_output_size = self.reference_output_dim
        metadata = self.reference_metadata if self.model_type in HETERO_MODEL_TYPES else None
        self.model = self._build_model(sample_data, metadata, per_node_output_size)

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
        accum_batches = 0
        step_start_time = None
        step_samples = 0

        def run_batch(case_idx, batch, batch_idx, total_steps, pbar, advance_pbar, include_case_name):
            nonlocal accum_batches, step_start_time, step_samples, total_loss, total_task_loss, num_batches
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

            should_step = ((batch_idx + 1) % self.accumulate_grad_batches == 0) or ((batch_idx + 1) == total_steps)

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

            if pbar is not None and self.global_rank == 0 and not self.wandb_enabled:
                postfix = {"loss": loss_value}
                if include_case_name:
                    postfix["case"] = self.case_names[case_idx]
                if self.log_every_n_steps and self.log_every_n_steps > 0:
                    if self.global_step % self.log_every_n_steps == 0:
                        pbar.set_postfix(postfix)
                else:
                    pbar.set_postfix(postfix)
                if advance_pbar:
                    pbar.update(1)

            if self._maybe_run_validation() and self.stop_training:
                return True
            self._maybe_save_periodic_checkpoint()
            if self._maybe_stop_by_samples():
                return True
            return False

        mix_every = self.case_mix_every_n_steps
        if mix_every <= 0 or len(self.train_case_indices) <= 1:
            for case_idx in self.train_case_indices:
                loader = self.train_loaders[case_idx]
                sampler = self.train_samplers.get(case_idx)
                if sampler is not None:
                    sampler.set_epoch(epoch)
                elif hasattr(loader.dataset, "set_epoch"):
                    loader.dataset.set_epoch(epoch)

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
                    should_break = run_batch(
                        case_idx,
                        batch,
                        batch_idx,
                        total_steps,
                        pbar,
                        advance_pbar=False,
                        include_case_name=False,
                    )
                    if should_break:
                        break

                if self.stop_training:
                    break
        else:
            for case_idx in self.train_case_indices:
                loader = self.train_loaders[case_idx]
                sampler = self.train_samplers.get(case_idx)
                if sampler is not None:
                    sampler.set_epoch(epoch)
                elif hasattr(loader.dataset, "set_epoch"):
                    loader.dataset.set_epoch(epoch)

            loader_lengths = {}
            active_cases = []
            total_steps = 0
            for case_idx in self.train_case_indices:
                steps = len(self.train_loaders[case_idx])
                loader_lengths[case_idx] = steps
                if steps > 0:
                    active_cases.append(case_idx)
                    total_steps += steps

            if total_steps == 0:
                return 0.0, 0.0, {}

            iterators = {case_idx: iter(self.train_loaders[case_idx]) for case_idx in active_cases}
            if self.global_rank == 0 and not self.wandb_enabled:
                pbar = tqdm(total=total_steps, desc=f"Epoch {epoch}")
            else:
                pbar = None

            self.optimizer.zero_grad()
            accum_batches = 0
            step_start_time = None
            step_samples = 0

            def iter_case_schedule():
                steps_left = {case_idx: loader_lengths[case_idx] for case_idx in active_cases}
                while True:
                    did_yield = False
                    for case_idx in active_cases:
                        remaining = steps_left[case_idx]
                        if remaining <= 0:
                            continue
                        take = mix_every if remaining > mix_every else remaining
                        for _ in range(take):
                            yield case_idx
                        steps_left[case_idx] = remaining - take
                        did_yield = True
                    if not did_yield:
                        break

            for batch_idx, case_idx in enumerate(iter_case_schedule()):
                batch = next(iterators[case_idx])
                should_break = run_batch(
                    case_idx,
                    batch,
                    batch_idx,
                    total_steps,
                    pbar,
                    advance_pbar=True,
                    include_case_name=True,
                )
                if should_break:
                    break

            if pbar is not None:
                pbar.close()

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

        if self.violation_eval_p <= 0.0:
            return None, None, None

        total_loss = 0.0
        total_task_loss = 0.0
        num_batches = 0
        metric_sums, metric_counts = self._init_metric_trackers(self.val_metric_names)
        timed_batches = 0

        with torch.no_grad():
            for case_idx in self.val_case_indices:
                loader = self.val_loaders[case_idx]
                if loader is None:
                    continue
                rng = self._make_violation_eval_rng(case_idx)
                try:
                    total_batches = len(loader)
                except Exception:
                    total_batches = None
                min_eval_batches = self.violation_eval_min_batches
                if total_batches is not None and total_batches > 0:
                    min_eval_batches = min(min_eval_batches, total_batches)
                eval_batches = 0

                for batch_idx, batch in enumerate(loader):
                    do_eval = self._should_eval_violations(
                        rng,
                        batch_idx,
                        total_batches,
                        eval_batches,
                        min_eval_batches,
                    )
                    if not do_eval:
                        continue
                    eval_batches += 1

                    do_timing = self._should_time_validation_batch(batch_idx, timed_batches)
                    if do_timing:
                        self._sync_for_timing()
                        timing_start = time.perf_counter()
                    batch = batch.to(self.device)
                    if do_timing:
                        self._sync_for_timing()
                        timing_after_data = time.perf_counter()

                    predictions = self.forward(batch)
                    if do_timing:
                        self._sync_for_timing()
                        timing_after_forward = time.perf_counter()
                    if self.loss_managers[case_idx].lagrangian is None:
                        loss, loss_info = self.loss_managers[case_idx].compute_loss(
                            predictions,
                            batch,
                            return_info=True,
                            collect_constraints=True,
                        )
                    else:
                        loss, loss_info = self.loss_managers[case_idx].compute_loss(
                            predictions,
                            batch,
                            return_info=True,
                        )
                    if do_timing:
                        self._sync_for_timing()
                        timing_after_loss = time.perf_counter()
                        self._add_metric(
                            metric_sums,
                            metric_counts,
                            "val/perf/data_ms",
                            (timing_after_data - timing_start) * 1000.0,
                        )
                        self._add_metric(
                            metric_sums,
                            metric_counts,
                            "val/perf/forward_ms",
                            (timing_after_forward - timing_after_data) * 1000.0,
                        )
                        self._add_metric(
                            metric_sums,
                            metric_counts,
                            "val/perf/loss_ms",
                            (timing_after_loss - timing_after_forward) * 1000.0,
                        )
                        self._add_metric(
                            metric_sums,
                            metric_counts,
                            "val/perf/total_ms",
                            (timing_after_loss - timing_start) * 1000.0,
                        )
                        timed_batches += 1

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
        eval_batches_tensor = torch.tensor(float(num_batches), device=self.device)
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(eval_batches_tensor, op=dist.ReduceOp.SUM)
        metric_avgs["val/perf/eval_batches"] = float(eval_batches_tensor.item())
        self._add_val_score(metric_avgs)

        return loss_tensor[0].item(), loss_tensor[1].item(), metric_avgs
