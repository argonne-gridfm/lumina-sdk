"""
Copyright (c) 2025 by Argonne National Laboratory.
All rights reserved.
"""
import os


def check_dir(dir_path):
    """Ensure a directory exists, creating it (and parents) if necessary.

    Args:
        dir_path (str): Path to the directory to verify or create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
