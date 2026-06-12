"""Heterogeneous OPF data preprocessing for Frontier (ROCm 7.1.1, no mpi4py).

Processes raw OPFDataset tar.gz files (downloaded from GCS) into heterogeneous
PyG HeteroData `.pt` files consumed by the HGT/RGAT/HEAT trainers.

Unlike data_process.py (which produces OPFOnDiskHomogeneousDataset for GCN/GAT),
this script produces OPFDataset (heterogeneous graph format) required by the
multi-relational models (HGT, RGAT, HEAT, HeteroGNN).

Output layout:
  <root>/OPFData/processed/dataset_release_1/<case_name>/group_<id>.pt

Usage (interactive or via srun):
  srun --ntasks=<N> python scripts/data_process_frontier.py

The script distributes (case_name, group_id) tasks round-robin across SLURM
tasks using SLURM_PROCID / SLURM_NTASKS. Falls back to a single process when
run outside SLURM.
"""

import os
from pathlib import Path

from lumina.dataset.opf.opf_dataset import OPFDataset


def _parse_int_list(raw, default):
    if raw is None or str(raw).strip() == "":
        return list(default)
    values = []
    for token in str(raw).replace(",", " ").split():
        values.append(int(token))
    return values


def _parse_str_list(raw, default):
    if raw is None or str(raw).strip() == "":
        return list(default)
    values = []
    for token in str(raw).replace(",", " ").split():
        token = token.strip()
        if token:
            values.append(token)
    return values


def get_rank_size():
    """Return (rank, world_size) from SLURM env vars (no mpi4py required)."""
    rank = int(
        os.environ.get("SLURM_PROCID")
        or os.environ.get("RANK")
        or 0
    )
    size = int(
        os.environ.get("SLURM_NTASKS")
        or os.environ.get("WORLD_SIZE")
        or 1
    )
    return rank, size


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    root = os.environ.get("LUMINA_PREPROCESS_ROOT", str(repo_root))
    default_groups = list(range(20))
    group_ids = _parse_int_list(os.environ.get("LUMINA_PREPROCESS_GROUP_IDS"), default_groups)
    default_cases = [
        "pglib_opf_case500_goc",
        "pglib_opf_case2000_goc",
    ]
    case_names = _parse_str_list(os.environ.get("LUMINA_PREPROCESS_CASES"), default_cases)

    case_mapping = {
        case_name: group_ids
        for case_name in case_names
    }

    tasks = [
        (case_name, group_id)
        for case_name, gids in case_mapping.items()
        for group_id in gids
    ]

    rank, size = get_rank_size()
    my_tasks = tasks[rank::size]

    print(f"[rank {rank}/{size}] root={root}")
    print(f"[rank {rank}/{size}] assigned {len(my_tasks)} task(s): {my_tasks}")

    for case_name, group_id in my_tasks:
        print(f"[rank {rank}] processing {case_name}, group_id={group_id} ...")
        OPFDataset(
            root=root,
            case_name=case_name,
            group_id=group_id,
            topological_perturbations=False,
            keep_temp=True,   # avoid race: tasks share gridopt-dataset-tmp; cleanup done by job script
            n_jobs=-1,
        )
        print(f"[rank {rank}] done: {case_name}, group_id={group_id}")
