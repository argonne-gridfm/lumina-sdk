module reset
ml cpe/24.07
ml cce/18.0.0

# --- ROCm 7.1 toolchain ---
ml rocm/7.1.1
ml amd-mixed/7.1.1

ml craype-accel-amd-gfx90a
ml PrgEnv-gnu
ml miniforge3/23.11.0-0
ml git-lfs
module unload darshan-runtime
