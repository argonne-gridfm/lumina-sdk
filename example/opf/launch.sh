export MASTER_ADDR=$(head -1 $PBS_NODEFILE)
export MASTER_PORT=29500

module use /soft/modulefiles
module load conda
conda activate lumina-core

mpiexec -n 8 -ppn 4 python train_opf_ddp.py