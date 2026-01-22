module reset
#module load gcc/12.3.0
ml cpe/24.07
ml cce/18.0.0
ml rocm/6.4.0
ml amd-mixed/6.4.0
ml craype-accel-amd-gfx90a
ml PrgEnv-gnu
ml miniforge3/23.11.0-0
#ml cmake/3.27.9
module unload darshan-runtime
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

