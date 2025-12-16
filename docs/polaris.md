## ALCF Polaris

To activate the shared `lumina-core` environment on Polaris:

```shell
module use /soft/modulefiles
module load conda
conda activate base

source /eagle/GridFM/conda_envs/lumina/bin/activate
```

This enables the centrally managed lumina-core Python environment.

Do not modify the shared environment; create a local virtual environment for experiments if needed.
