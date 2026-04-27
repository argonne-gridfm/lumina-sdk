import os
import os.path as osp
import time
from contextlib import contextmanager
from typing import Optional

from torch_geometric.io import fs

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def _expand_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    expanded = osp.expanduser(os.path.expandvars(path))
    if "$" in expanded:
        return None
    return expanded


def resolve_stage_root(staging_config: Optional[dict]) -> Optional[str]:
    root = None
    if isinstance(staging_config, dict):
        root = staging_config.get("root")
    root = _expand_path(root)
    if root:
        return root
    for env_key in ("LUMINA_STAGE_ROOT", "SLURM_TMPDIR", "TMPDIR"):
        env_root = os.environ.get(env_key)
        if env_root:
            return _expand_path(osp.join(env_root, "lumina_stage"))
    return None


def opf_release(processed_suffix: Optional[str] = None) -> str:
    release = "dataset_release_1"
    if processed_suffix:
        release += f"_{processed_suffix}"
    return release


def on_disk_processed_dir(
    root: str,
    case_name: str,
    processed_suffix: Optional[str] = None,
) -> str:
    return osp.join(
        root,
        "OPFData",
        "on_disk",
        opf_release(processed_suffix),
        case_name,
    )


def on_disk_db_name(group_id: int, backend: str) -> str:
    if backend == "rocksdb":
        return f"group_{group_id}.rocksdb"
    return f"group_{group_id}.{backend}.db"


def get_on_disk_db_path(
    root: str,
    case_name: str,
    group_id: int,
    backend: str,
    processed_suffix: Optional[str] = None,
) -> str:
    processed_dir = on_disk_processed_dir(root, case_name, processed_suffix)
    return osp.join(processed_dir, on_disk_db_name(group_id, backend))


def get_on_disk_lock_path(
    root: str,
    case_name: str,
    group_id: int,
    backend: str,
    processed_suffix: Optional[str] = None,
) -> str:
    return (
        get_on_disk_db_path(
            root,
            case_name,
            group_id,
            backend,
            processed_suffix,
        )
        + ".lock"
    )


def sharded_processed_dir(
    root: str,
    case_name: str,
    processed_suffix: Optional[str] = None,
) -> str:
    return osp.join(
        root,
        "OPFData",
        "sharded",
        opf_release(processed_suffix),
        case_name,
    )


def get_sharded_manifest_path(
    root: str,
    case_name: str,
    processed_suffix: Optional[str] = None,
    manifest_name: str = "manifest.json",
) -> str:
    return osp.join(
        sharded_processed_dir(root, case_name, processed_suffix),
        manifest_name,
    )


def get_sharded_lock_path(
    root: str,
    case_name: str,
    processed_suffix: Optional[str] = None,
    manifest_name: str = "manifest.json",
) -> str:
    return (
        get_sharded_manifest_path(
            root,
            case_name,
            processed_suffix,
            manifest_name,
        )
        + ".lock"
    )


@contextmanager
def file_lock(path: str, timeout_sec: Optional[int] = 3600):
    if fcntl is None:
        yield
        return

    lock_dir = osp.dirname(path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    start = time.time()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if timeout_sec is not None and (time.time() - start) > timeout_sec:
                    raise TimeoutError(f"Timed out waiting for lock: {path}")
                time.sleep(1)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _same_size(path_a: str, path_b: str) -> bool:
    try:
        info_a = fs.info(path_a)
        info_b = fs.info(path_b)
        if "size" in info_a and "size" in info_b:
            return info_a["size"] == info_b["size"]
    except Exception:
        return False
    return False


def _copy_sidecar_files(src_path: str, dst_path: str, log: bool) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        src_sidecar = f"{src_path}{suffix}"
        if fs.exists(src_sidecar):
            fs.cp(src_sidecar, f"{dst_path}{suffix}", log=log)


def stage_on_disk_group(
    source_root: str,
    stage_root: str,
    case_name: str,
    group_id: int,
    backend: str,
    processed_suffix: Optional[str] = None,
    log: bool = True,
) -> str:
    source_root = _expand_path(source_root) or source_root
    stage_root = _expand_path(stage_root) or stage_root

    if not stage_root or not source_root:
        return source_root

    if osp.abspath(stage_root) == osp.abspath(source_root):
        return source_root

    src_path = get_on_disk_db_path(
        source_root,
        case_name,
        group_id,
        backend,
        processed_suffix,
    )
    dst_path = get_on_disk_db_path(
        stage_root,
        case_name,
        group_id,
        backend,
        processed_suffix,
    )

    if not fs.exists(src_path):
        raise FileNotFoundError(f"On-disk DB missing at {src_path}")

    if fs.exists(dst_path):
        if not fs.isdir(src_path) and _same_size(src_path, dst_path):
            return stage_root

    dst_dir = osp.dirname(dst_path)
    fs.makedirs(dst_dir, exist_ok=True)

    if fs.isdir(src_path):
        if fs.exists(dst_path):
            fs.rm(dst_path, recursive=True)
        fs.cp(src_path, dst_dir, log=log)
    else:
        fs.cp(src_path, dst_path, log=log)
        _copy_sidecar_files(src_path, dst_path, log=log)

    return stage_root


def stage_sharded_case(
    source_root: str,
    stage_root: str,
    case_name: str,
    processed_suffix: Optional[str] = None,
    manifest_name: str = "manifest.json",
    log: bool = True,
) -> str:
    source_root = _expand_path(source_root) or source_root
    stage_root = _expand_path(stage_root) or stage_root

    if not stage_root or not source_root:
        return source_root

    if osp.abspath(stage_root) == osp.abspath(source_root):
        return source_root

    src_dir = sharded_processed_dir(source_root, case_name, processed_suffix)
    dst_dir = sharded_processed_dir(stage_root, case_name, processed_suffix)
    src_manifest = get_sharded_manifest_path(
        source_root,
        case_name,
        processed_suffix,
        manifest_name,
    )
    dst_manifest = get_sharded_manifest_path(
        stage_root,
        case_name,
        processed_suffix,
        manifest_name,
    )

    if not fs.exists(src_dir):
        raise FileNotFoundError(f"Sharded dataset missing at {src_dir}")

    if fs.exists(dst_manifest) and _same_size(src_manifest, dst_manifest):
        return stage_root

    dst_parent = osp.dirname(dst_dir)
    fs.makedirs(dst_parent, exist_ok=True)
    if fs.exists(dst_dir):
        fs.rm(dst_dir, recursive=True)
    fs.cp(src_dir, dst_parent, log=log)
    return stage_root
