import argparse
import json
import os
import os.path as osp
import random
import shutil
import sys

import torch
from torch_geometric.data import InMemoryDataset

from lumina.dataset.opf.staging import opf_release


def _num_samples_from_slices(slices) -> int:
    if torch.is_tensor(slices):
        return int(slices.numel()) - 1
    if isinstance(slices, dict):
        for value in slices.values():
            return _num_samples_from_slices(value)
    raise ValueError("Unable to infer sample count from slices.")


class _InMemoryShardAccessor:
    def __init__(self, data, slices):
        self._data = data
        self.slices = slices

    @property
    def data(self):
        return self._data

    def get(self, idx: int):
        return InMemoryDataset.get(self, idx)

    def len(self) -> int:
        return _num_samples_from_slices(self.slices)

    def __len__(self) -> int:
        return self.len()


def _iter_samples(obj):
    if isinstance(obj, tuple) and len(obj) == 2:
        data, slices = obj
        num_samples = _num_samples_from_slices(slices)
        accessor = _InMemoryShardAccessor(data, slices)
        for idx in range(num_samples):
            yield accessor.get(idx)
        return
    if isinstance(obj, list):
        for sample in obj:
            yield sample
        return
    raise ValueError("Unsupported shard payload type.")


def _shard_len(obj) -> int:
    if isinstance(obj, tuple) and len(obj) == 2:
        _, slices = obj
        return _num_samples_from_slices(slices)
    if isinstance(obj, list):
        return len(obj)
    raise ValueError("Unsupported shard payload type.")


def _link_or_copy(src: str, dst: str, mode: str, overwrite: bool) -> None:
    if osp.exists(dst):
        if overwrite:
            os.remove(dst)
        else:
            raise FileExistsError(f"Shard already exists: {dst}")

    os.makedirs(osp.dirname(dst), exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            mode = "copy"
    if mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def _assign_splits(shards, train_ratio, val_ratio, seed, shuffle):
    if not shards:
        return {"train": [], "val": [], "test": []}
    shards = list(shards)
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(shards)
    total = sum(shard["num_samples"] for shard in shards)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)
    splits = {"train": [], "val": [], "test": []}
    seen = 0
    for shard in shards:
        if seen < train_target:
            splits["train"].append(shard["name"])
        elif seen < train_target + val_target:
            splits["val"].append(shard["name"])
        else:
            splits["test"].append(shard["name"])
        seen += shard["num_samples"]
    if not splits["train"]:
        splits["train"].append(shards[0]["name"])
        splits["test"] = [entry["name"] for entry in shards[1:]]
    return splits


def parse_args():
    parser = argparse.ArgumentParser(description="Build sharded OPF datasets from processed .pt files.")
    parser.add_argument("--root", type=str, required=True, help="Dataset root (same as training config root).")
    parser.add_argument("--case-name", type=str, required=True, help="Case name (pglib_opf_*).")
    parser.add_argument("--group-ids", type=int, nargs="+", required=True, help="Group IDs to shard.")
    parser.add_argument(
        "--processed-suffix",
        type=str,
        default=None,
        help="Processed suffix (e.g. 'homo' for precomputed homogeneous).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root to write sharded data (default: --root).",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="manifest.json",
        help="Manifest file name inside sharded directory.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help="Samples per shard (<=0 keeps each group file as a shard).",
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        choices=["copy", "hardlink", "symlink"],
        default="hardlink",
        help="How to place group shards in sharded directory when not re-sharding.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only write manifest referencing source files; do not copy/link shards.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard files.",
    )
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for split assignment.")
    parser.add_argument(
        "--no-split-shuffle",
        action="store_true",
        help="Disable shuffling before split assignment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    release = opf_release(args.processed_suffix)
    processed_dir = osp.join(args.root, "OPFData", "processed", release, args.case_name)
    output_root = args.output_root or args.root
    sharded_dir = osp.join(output_root, "OPFData", "sharded", release, args.case_name)
    os.makedirs(sharded_dir, exist_ok=True)

    shard_size = int(args.shard_size)
    shards = []
    total_samples = 0

    for group_id in args.group_ids:
        src_path = osp.join(processed_dir, f"group_{group_id}.pt")
        if not osp.exists(src_path):
            raise FileNotFoundError(f"Missing processed group file: {src_path}")

        obj = torch.load(src_path, map_location="cpu", weights_only=False)
        group_samples = _shard_len(obj)
        total_samples += int(group_samples)

        if shard_size <= 0 or shard_size >= group_samples:
            shard_name = f"group_{group_id}.pt"
            if args.manifest_only:
                shard_path = src_path
            else:
                shard_path = osp.join(sharded_dir, shard_name)
                _link_or_copy(src_path, shard_path, args.link_mode, args.overwrite)
                shard_path = shard_name
            shards.append(
                {
                    "name": shard_name,
                    "path": shard_path,
                    "num_samples": int(group_samples),
                    "group_id": int(group_id),
                }
            )
            continue

        shard_idx = 0
        buffer = []
        for sample in _iter_samples(obj):
            buffer.append(sample)
            if len(buffer) >= shard_size:
                shard_name = f"group_{group_id}_shard_{shard_idx:05d}.pt"
                shard_path = osp.join(sharded_dir, shard_name)
                if not args.overwrite and osp.exists(shard_path):
                    raise FileExistsError(f"Shard already exists: {shard_path}")
                data, slices = InMemoryDataset.collate(buffer)
                torch.save((data, slices), shard_path)
                shards.append(
                    {
                        "name": shard_name,
                        "path": shard_name,
                        "num_samples": len(buffer),
                        "group_id": int(group_id),
                    }
                )
                buffer = []
                shard_idx += 1

        if buffer:
            shard_name = f"group_{group_id}_shard_{shard_idx:05d}.pt"
            shard_path = osp.join(sharded_dir, shard_name)
            if not args.overwrite and osp.exists(shard_path):
                raise FileExistsError(f"Shard already exists: {shard_path}")
            data, slices = InMemoryDataset.collate(buffer)
            torch.save((data, slices), shard_path)
            shards.append(
                {
                    "name": shard_name,
                    "path": shard_name,
                    "num_samples": len(buffer),
                    "group_id": int(group_id),
                }
            )

    splits = _assign_splits(
        shards,
        train_ratio=float(args.train_split),
        val_ratio=float(args.val_split),
        seed=int(args.split_seed),
        shuffle=not args.no_split_shuffle,
    )

    manifest = {
        "format": "pt_inmemory_v1",
        "case_name": args.case_name,
        "release": release,
        "processed_suffix": args.processed_suffix,
        "shard_size": shard_size if shard_size > 0 else None,
        "num_samples": total_samples,
        "shards": shards,
        "splits": splits,
    }
    if not args.manifest_only:
        manifest["base_dir"] = "."

    manifest_path = osp.join(sharded_dir, args.manifest_name)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Wrote shard manifest to {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Failed to build shards: {exc}", file=sys.stderr)
        raise
