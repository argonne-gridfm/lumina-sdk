"""Minimal single-process OPF training example.

Trains :class:`OPFHeteroGNN` on a single OPF case using settings from
``configs/config.yaml`` (loader, optimizer, training) and
``configs/model/heterognn.yaml`` (model architecture). No DDP, MPI, or
``torchrun`` involved — runs as a plain Python process. Intended as a smoke
test on a single workstation.

Run:
    python example/opf/train_opf_simple.py
"""
import argparse
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.dataset.opf.transforms import to_float32
from lumina.loader.opf.opf_loader import DataLoader
from lumina.model.opf.hetero_model import OPFHeteroGNN
from lumina.model.opf.losses import OPFLossManager


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(repo_root / "configs" / "config.yaml"),
                        help="Main YAML config (loader / optimizer / training).")
    parser.add_argument("--model_config",
                        default=str(repo_root / "configs" / "model" / "heterognn.yaml"),
                        help="Model architecture YAML.")
    parser.add_argument("--case", default="pglib_opf_case14_ieee",
                        help="pglib-opf case name.")
    parser.add_argument("--group_id", type=int, default=0,
                        help="Dataset group (each group ~ 15k samples).")
    parser.add_argument("--device", default=None,
                        help="Override device (default: cuda if available else cpu).")
    return parser.parse_args()


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)["models"]["HeteroGNN"]

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    dataset = OPFDataset(
        root=cfg["root"],
        case_name=args.case,
        group_id=args.group_id,
        transform=to_float32,
    )
    print(f"Loaded {len(dataset)} samples from {args.case} group {args.group_id}")

    train_frac = cfg.get("train_split", 0.8)
    val_frac = cfg.get("val_split", 0.1)
    n_total = len(dataset)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    loader_cfg = cfg.get("loader", {})
    batch_size = loader_cfg.get("batch_size", 16)
    num_workers = loader_cfg.get("num_workers", 0)

    train_loader = DataLoader(dataset[:n_train], batch_size=batch_size,
                              shuffle=loader_cfg.get("shuffle", True),
                              num_workers=num_workers)
    val_loader = DataLoader(dataset[n_train:n_train + n_val],
                            batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    sample = dataset[0]
    input_channels = {nt: sample[nt].x.size(-1) for nt in sample.node_types}
    model = OPFHeteroGNN(
        metadata=sample.metadata(),
        input_channels=input_channels,
        hidden_channels=model_cfg.get("hidden_channels", 64),
        num_layers=model_cfg.get("num_layers", 3),
        backend=model_cfg.get("backend", "sage"),
    ).to(device)

    loss_manager = OPFLossManager(loss_type="mse")
    optim_cfg = cfg["optimizer"]["AdamW"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg["lr"],
        betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
        eps=optim_cfg.get("eps", 1e-8),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
    )

    max_epochs = cfg["training"]["max_epochs"]
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = 0.0
        train_pbar = tqdm(train_loader, desc=f"epoch {epoch}/{max_epochs} [train]",
                          leave=False, dynamic_ncols=True)
        for step, batch in enumerate(train_pbar, start=1):
            batch = batch.to(device)
            pred = model(batch.x_dict, batch.edge_index_dict)
            loss, _ = loss_manager.compute_loss(pred, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item()
            train_pbar.set_postfix(loss=f"{train_total / step:.4f}")
        train_avg = train_total / len(train_loader)

        model.eval()
        val_total = 0.0
        val_pbar = tqdm(val_loader, desc=f"epoch {epoch}/{max_epochs} [val]  ",
                        leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for step, batch in enumerate(val_pbar, start=1):
                batch = batch.to(device)
                pred = model(batch.x_dict, batch.edge_index_dict)
                loss, _ = loss_manager.compute_loss(pred, batch)
                val_total += loss.item()
                val_pbar.set_postfix(loss=f"{val_total / step:.4f}")
        val_avg = val_total / len(val_loader)

        tqdm.write(
            f"epoch {epoch:2d}/{max_epochs} | "
            f"train_loss={train_avg:.4f} | val_loss={val_avg:.4f}"
        )


if __name__ == "__main__":
    main()
