## NERSC Perlmutter

To activate the shared `lumina-core` environment on Perlmutter:

```shell
module load conda

conda activate ${CFS}/amsc004/conda_envs/lumina
```

This enables the centrally managed lumina-core Python environment.

Do not modify the shared environment; create a local virtual environment for experiments if needed.
