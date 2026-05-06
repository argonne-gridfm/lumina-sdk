"""Minimal GPU test for Frontier ROCm venv — run via test_gpu_frontier.sh"""
import os
import sys

rank = int(os.environ.get("SLURM_PROCID", 0))
local_rank = int(os.environ.get("SLURM_LOCALID", 0))

print(f"[rank {rank}] Python {sys.version}", flush=True)
print(f"[rank {rank}] ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES','unset')}", flush=True)

import torch
print(f"[rank {rank}] torch {torch.__version__}", flush=True)
print(f"[rank {rank}] torch.cuda.is_available() = {torch.cuda.is_available()}", flush=True)
print(f"[rank {rank}] torch.cuda.device_count() = {torch.cuda.device_count()}", flush=True)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    print(f"[rank {rank}] GPU 0 name: {name}", flush=True)
    t = torch.randn(4, 4, device="cuda")
    s = t.sum()
    print(f"[rank {rank}] randn + sum OK: {s.item():.4f}", flush=True)
    # Test a simple all-reduce
    import torch.distributed as dist
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.all_reduce(t)
    dist.destroy_process_group()
    print(f"[rank {rank}] all_reduce OK", flush=True)
else:
    print(f"[rank {rank}] ERROR: no CUDA/HIP device visible", flush=True)
    sys.exit(1)

print(f"[rank {rank}] ALL TESTS PASSED", flush=True)
