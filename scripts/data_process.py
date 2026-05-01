"""Bulk pre-process OPFData groups across all PGLib cases.

Splits the (case, group_id) work list across MPI ranks so a multi-process
launcher (`mpirun`/`srun`) can fan out the dataset construction. Each rank
materialises its share of `OPFOnDiskHomogeneousDataset` instances under
``--root``.

Usage:
    python scripts/data_process.py --root /path/to/dataset/cache
    mpirun -np 8 python scripts/data_process.py --root /path/to/dataset/cache
"""
import argparse
import os

from lumina.dataset.opf.opf_dataset import OPFDataset, OPFHomogeneousDataset
from lumina.dataset.opf.opf_on_disk_dataset import OPFOnDiskDataset, OPFOnDiskHomogeneousDataset


def get_rank_size():
    try:
        from mpi4py import MPI
    except Exception:
        MPI = None

    if MPI is not None:
        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()

    # Fallback to common launcher environment variables when mpi4py is unavailable.
    env = os.environ
    rank = env.get("OMPI_COMM_WORLD_RANK") or env.get("PMI_RANK") or env.get("SLURM_PROCID")
    rank = rank or env.get("MV2_COMM_WORLD_RANK") or env.get("RANK") or "0"
    size = env.get("OMPI_COMM_WORLD_SIZE") or env.get("PMI_SIZE") or env.get("SLURM_NTASKS")
    size = size or env.get("MV2_COMM_WORLD_SIZE") or env.get("WORLD_SIZE") or "1"
    return int(rank), int(size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="./opf_data",
        help="Dataset root directory (default: ./opf_data).",
    )
    args = parser.parse_args()

    group_ids = [0, 1]  # extend as needed, e.g. list(range(20))
    case_mapping = {
        "pglib_opf_case14_ieee": group_ids,
        "pglib_opf_case30_ieee": group_ids,
        "pglib_opf_case57_ieee": group_ids,
        "pglib_opf_case118_ieee": group_ids,
        'pglib_opf_case500_goc': group_ids,
        'pglib_opf_case2000_goc': group_ids,
        'pglib_opf_case4661_sdet': group_ids,
        'pglib_opf_case6470_rte': group_ids,
        'pglib_opf_case10000_goc': group_ids,
        'pglib_opf_case13659_pegase': group_ids,
    }

    tasks = [(case_name, group_id)
             for case_name, group_ids in case_mapping.items()
             for group_id in group_ids]
    rank, size = get_rank_size()
    for case_name, group_id in tasks[rank::size]:
        print(f"Processing case [rank: {rank}]: {case_name}, group_id: {group_id}")
        # OPFDataset(root=args.root, case_name=case_name, group_id=group_id)
        # OPFHomogeneousDataset(root=args.root, case_name=case_name, group_id=group_id)
        # OPFOnDiskDataset(root=args.root, case_name=case_name, group_id=group_id)
        OPFOnDiskHomogeneousDataset(root=args.root, case_name=case_name, group_id=group_id)
