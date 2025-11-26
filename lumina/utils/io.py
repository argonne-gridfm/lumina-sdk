import os


def check_dir(dir_path):
    r""" Check if the directory exists, if not create it.

    Args:
        dir_path (str): Directory path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
