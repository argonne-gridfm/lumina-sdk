#!/bin/bash
# Per-rank launcher for Frontier (ROCm/HIP).
# Sets ROCR_VISIBLE_DEVICES to the node-local GPU index so each process sees
# exactly one GPU at HIP device index 0.
export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}
exec "$@"
