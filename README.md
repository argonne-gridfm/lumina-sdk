# LUMINA - A Large-scale Unified Model for Intelligent Grid Applications

Lumina Core is the core package for LUMINA, a large-scale unified model for intelligent grid applications. It provides essential functionalities for data processing, model architecture, evaluator and more. 

## Install Instructions for Lumina Core

1. create and activate a virtual environment (recommended)

```
python -m venv .venv
source .venv/bin/activate
```

2. install Lumina Core package and optional dependencies

- install the general package in editable mode
```
pip install -e . 
```

- install optional dependencies as needed, e.g., for cuGraph support
```
pip install -e .[cuGraph] -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

- checkout `Makefile` for other optional dependencies installation commands.
```
make help
```

## LICENSE

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.