#!/bin/bash
# Sync offline W&B logs from compute nodes to W&B servers
# Run this on the login node after SLURM jobs complete

WANDB_DIR="${1:-./wandb}"
PROJECT_ROOT="${2:-$(pwd)}"

echo "🔄 Syncing offline W&B logs from ${WANDB_DIR}..."

# Find all offline run directories
find "${WANDB_DIR}" -name "offline-run-*" -type d | while read offline_dir; do
    echo "📤 Syncing: ${offline_dir}"
    wandb sync "${offline_dir}" --no-include-synced
done

echo "✅ Sync complete!"
