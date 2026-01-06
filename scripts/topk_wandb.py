#!/usr/bin/env python3
"""
Minimal W&B API script: compute "best-after-burn-in" and extract top-k runs.

Ranking metric:
  rank = mean of the best K_BEST values of `metric_name` observed at steps >= burn_in_step
        (use K_BEST=1 for pure min; 2 or 3 is more robust)

Usage:
  python topk_wandb.py --entity <ENTITY> --project <PROJECT> --sweep <SWEEP_ID> \
    --metric val/score --step _step --burn 600000 --topk 5 --kbest 2

Notes:
- Requires: pip install wandb
- Assumes you logged metrics with: wandb.log({...}, step=...)
- Use --step _step for the W&B step axis; use a metric key only if you logged it explicitly.
- Defaults to reading history artifacts (parquet) with API fallback.
- Artifact history requires pyarrow or pandas.
- Use --inspect-run <id> to list available columns and sample stats.
"""

import argparse
import heapq
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import wandb


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def extract_run_id(value: str) -> str:
    match = re.search(r"run-([A-Za-z0-9]+)-history", value)
    if match:
        return match.group(1)
    return value


def build_artifact_full_name(
    entity: str,
    project: str,
    run_id: str,
    artifact_name: str,
    artifact_version: str,
) -> str:
    try:
        base_name = artifact_name.format(id=run_id)
    except Exception:
        base_name = artifact_name
    if "/" in base_name:
        full_name = base_name
    else:
        full_name = f"{entity}/{project}/{base_name}"
    if ":" not in full_name:
        full_name = f"{full_name}:{artifact_version}"
    return full_name


def download_artifact_dir(artifact: Any, run_dir: Path) -> Optional[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        try:
            return Path(artifact.download(root=str(run_dir), replace=False))
        except TypeError:
            return Path(artifact.download(root=str(run_dir)))
    except Exception:
        return None


def find_parquet_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.stat().st_size > 0 else []
    return [path for path in sorted(root.rglob("*.parquet")) if path.stat().st_size > 0]


def get_parquet_files_for_run(
    api: wandb.Api,
    entity: str,
    project: str,
    run_id: str,
    cache_dir: Path,
    artifact_name: str,
    artifact_version: str,
    artifact_type: str,
) -> Optional[List[Path]]:
    run_dir = cache_dir / run_id / "artifact"
    cached = find_parquet_files(run_dir)
    if cached:
        return cached

    full_name = build_artifact_full_name(
        entity=entity,
        project=project,
        run_id=run_id,
        artifact_name=artifact_name,
        artifact_version=artifact_version,
    )
    try:
        if hasattr(api, "use_artifact"):
            artifact = api.use_artifact(full_name, type=artifact_type)
        else:
            artifact = api.artifact(full_name, type=artifact_type)
    except Exception:
        return None

    artifact_dir = download_artifact_dir(artifact, run_dir)
    if artifact_dir is None:
        return None
    files = find_parquet_files(artifact_dir)
    return files or None


def iter_parquet_rows(
    files: Sequence[Path],
    step_name: str,
    metric_name: str,
) -> Iterator[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception:
        yield from iter_parquet_rows_pandas(files, step_name, metric_name)
        return

    for path in files:
        try:
            pf = pq.ParquetFile(path)
        except Exception:
            continue
        schema_names = set(pf.schema.names)
        if step_name not in schema_names or metric_name not in schema_names:
            continue
        for batch in pf.iter_batches(columns=[step_name, metric_name]):
            data = batch.to_pydict()
            steps = data.get(step_name)
            metrics = data.get(metric_name)
            if steps is None or metrics is None:
                continue
            for step, metric in zip(steps, metrics):
                yield {step_name: step, metric_name: metric}


def iter_parquet_rows_pandas(
    files: Sequence[Path],
    step_name: str,
    metric_name: str,
) -> Iterator[Dict[str, Any]]:
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Reading parquet requires pyarrow or pandas.") from exc

    for path in files:
        try:
            df = pd.read_parquet(path, columns=[step_name, metric_name])
        except KeyError:
            continue
        except Exception as exc:
            msg = str(exc).lower()
            if "engine" in msg or "pyarrow" in msg or "fastparquet" in msg:
                raise RuntimeError("Reading parquet requires pyarrow or fastparquet.") from exc
            if "column" in msg or "columns" in msg or "not in index" in msg:
                continue
            raise

        for step, metric in df.itertuples(index=False, name=None):
            yield {step_name: step, metric_name: metric}


def iter_api_rows(
    run: wandb.apis.public.Run,
    step_name: str,
    metric_name: str,
) -> Iterable[Dict[str, Any]]:
    return run.scan_history(keys=[step_name, metric_name])


def best_after_burnin(
    rows: Iterable[Dict[str, Any]],
    metric_name: str,
    step_name: str,
    burn_in_step: float,
    kbest: int = 1,
) -> Tuple[Optional[float], int]:
    kbest = max(1, int(kbest))
    best_heap: List[float] = []
    npts = 0

    for row in rows:
        step = safe_float(row.get(step_name))
        metric = safe_float(row.get(metric_name))
        if step is None or metric is None:
            continue
        if step >= burn_in_step:
            npts += 1
            if len(best_heap) < kbest:
                heapq.heappush(best_heap, -metric)
            elif metric < -best_heap[0]:
                heapq.heapreplace(best_heap, -metric)

    if not best_heap:
        return None, 0

    best_vals = [-v for v in best_heap]
    return float(sum(best_vals) / len(best_vals)), npts


def summarize_rows(
    rows: Iterable[Dict[str, Any]],
    step_name: str,
    metric_name: str,
    max_rows: int = 5000,
) -> Dict[str, Optional[float]]:
    n_rows = 0
    n_step = 0
    n_metric = 0
    step_min = None
    step_max = None
    metric_min = None
    metric_max = None

    for row in rows:
        n_rows += 1
        step = safe_float(row.get(step_name))
        metric = safe_float(row.get(metric_name))
        if step is not None:
            n_step += 1
            step_min = step if step_min is None or step < step_min else step_min
            step_max = step if step_max is None or step > step_max else step_max
        if metric is not None:
            n_metric += 1
            metric_min = metric if metric_min is None or metric < metric_min else metric_min
            metric_max = metric if metric_max is None or metric > metric_max else metric_max
        if n_rows >= max_rows:
            break

    return {
        "rows": float(n_rows),
        "step_nonnull": float(n_step),
        "metric_nonnull": float(n_metric),
        "step_min": step_min,
        "step_max": step_max,
        "metric_min": metric_min,
        "metric_max": metric_max,
    }


def format_stat(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


def collect_parquet_columns(files: Sequence[Path], max_files: int = 5) -> Optional[List[str]]:
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None

    cols: Set[str] = set()
    for path in list(files)[:max_files]:
        try:
            pf = pq.ParquetFile(path)
        except Exception:
            continue
        cols.update(pf.schema.names)
    return sorted(cols)


def inspect_run_history(
    run: wandb.apis.public.Run,
    api: wandb.Api,
    args: argparse.Namespace,
    cache_dir: Path,
    use_artifact: bool,
    use_api: bool,
) -> None:
    print(f"Inspecting run id={run.id} name={run.name} state={run.state}")
    print(f"URL: {run.url}")

    if use_artifact:
        full_name = build_artifact_full_name(
            entity=args.entity,
            project=args.project,
            run_id=run.id,
            artifact_name=args.history_artifact,
            artifact_version=args.history_artifact_version,
        )
        print(f"Artifact name: {full_name} (type={args.history_artifact_type})")
        files = get_parquet_files_for_run(
            api=api,
            entity=args.entity,
            project=args.project,
            run_id=run.id,
            cache_dir=cache_dir,
            artifact_name=args.history_artifact,
            artifact_version=args.history_artifact_version,
            artifact_type=args.history_artifact_type,
        )
        if not files:
            print("No parquet files found for artifact.")
        else:
            cols = collect_parquet_columns(files)
            if not cols:
                print("No columns found. Install pyarrow to inspect parquet schema.")
            else:
                print(f"Parquet columns ({len(cols)}): {', '.join(cols[:50])}")
                if args.step not in cols or args.metric not in cols:
                    print(f"Missing columns: step={args.step in cols} metric={args.metric in cols}")
            try:
                stats = summarize_rows(
                    iter_parquet_rows(files, step_name=args.step, metric_name=args.metric),
                    step_name=args.step,
                    metric_name=args.metric,
                )
                print(
                    "Parquet sample rows={rows} step_nonnull={step_nonnull} metric_nonnull={metric_nonnull} "
                    "step_min={step_min} step_max={step_max} metric_min={metric_min} metric_max={metric_max}".format(
                        rows=format_stat(stats["rows"]),
                        step_nonnull=format_stat(stats["step_nonnull"]),
                        metric_nonnull=format_stat(stats["metric_nonnull"]),
                        step_min=format_stat(stats["step_min"]),
                        step_max=format_stat(stats["step_max"]),
                        metric_min=format_stat(stats["metric_min"]),
                        metric_max=format_stat(stats["metric_max"]),
                    )
                )
            except RuntimeError as exc:
                print(f"Parquet read failed: {exc}")

    if use_api and not use_artifact:
        stats = summarize_rows(
            iter_api_rows(run, step_name=args.step, metric_name=args.metric),
            step_name=args.step,
            metric_name=args.metric,
        )
        print(
            "API sample rows={rows} step_nonnull={step_nonnull} metric_nonnull={metric_nonnull} "
            "step_min={step_min} step_max={step_max} metric_min={metric_min} metric_max={metric_max}".format(
                rows=format_stat(stats["rows"]),
                step_nonnull=format_stat(stats["step_nonnull"]),
                metric_nonnull=format_stat(stats["metric_nonnull"]),
                step_min=format_stat(stats["step_min"]),
                step_max=format_stat(stats["step_max"]),
                metric_min=format_stat(stats["metric_min"]),
                metric_max=format_stat(stats["metric_max"]),
            )
        )


def score_run(
    run: wandb.apis.public.Run,
    api: wandb.Api,
    args: argparse.Namespace,
    cache_dir: Path,
    use_artifact: bool,
    use_api: bool,
) -> Tuple[Optional[float], int, Optional[str]]:
    if use_artifact:
        files = get_parquet_files_for_run(
            api=api,
            entity=args.entity,
            project=args.project,
            run_id=run.id,
            cache_dir=cache_dir,
            artifact_name=args.history_artifact,
            artifact_version=args.history_artifact_version,
            artifact_type=args.history_artifact_type,
        )
        if files:
            try:
                rank_val, npts = best_after_burnin(
                    iter_parquet_rows(files, step_name=args.step, metric_name=args.metric),
                    metric_name=args.metric,
                    step_name=args.step,
                    burn_in_step=args.burn,
                    kbest=args.kbest,
                )
            except RuntimeError as exc:
                if args.history_source == "artifact":
                    raise
                print(f"History artifact read failed for run {run.id}: {exc}", file=sys.stderr)
            else:
                if rank_val is not None:
                    return rank_val, npts, "artifact"

    if use_api:
        rank_val, npts = best_after_burnin(
            iter_api_rows(run, step_name=args.step, metric_name=args.metric),
            metric_name=args.metric,
            step_name=args.step,
            burn_in_step=args.burn,
            kbest=args.kbest,
        )
        if rank_val is not None:
            return rank_val, npts, "api"

    return None, 0, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True, help="W&B entity (user or team)")
    ap.add_argument("--project", required=True, help="W&B project name")
    ap.add_argument("--sweep", required=True, help="Sweep ID or path 'entity/project/sweep_id'")
    ap.add_argument("--metric", default="val/score", help="Metric name to rank by (lower is better)")
    ap.add_argument("--step", default="_step", help="History key used for burn-in (e.g., _step)")
    ap.add_argument("--burn", type=float, default=600000, help="Burn-in threshold on step axis")
    ap.add_argument("--topk", type=int, default=3, help="Number of top runs to output")
    ap.add_argument("--kbest", type=int, default=2, help="Mean of best k values after burn-in")
    ap.add_argument(
        "--state",
        default="all",
        help="Run state filter: finished|running|crashed|failed|killed|all",
    )
    ap.add_argument(
        "--history-source",
        default="auto",
        choices=("auto", "artifact", "api"),
        help="History source: auto tries artifacts then API scan_history",
    )
    ap.add_argument(
        "--history-artifact",
        default="run-{id}-history",
        help="History artifact base name or full path; use {id} for run id",
    )
    ap.add_argument(
        "--history-artifact-version",
        default="v0",
        help="History artifact version or alias (e.g., v0 or latest)",
    )
    ap.add_argument(
        "--history-artifact-type",
        default="wandb-history",
        help="History artifact type",
    )
    ap.add_argument(
        "--cache-dir",
        default=".wandb_history_cache",
        help="Cache directory for downloaded history artifacts",
    )
    ap.add_argument(
        "--inspect-run",
        default=None,
        help="Run id (or full artifact name containing run-<id>-history) to inspect history columns",
    )
    args = ap.parse_args()

    api = wandb.Api()

    if "/" in args.sweep:
        sweep_path = args.sweep
    else:
        sweep_path = f"{args.entity}/{args.project}/{args.sweep}"

    sweep = api.sweep(sweep_path)
    runs = sweep.runs

    cache_dir = Path(args.cache_dir).expanduser()
    use_artifact = args.history_source in ("auto", "artifact")
    use_api = args.history_source in ("auto", "api")

    if args.inspect_run:
        inspect_id = extract_run_id(args.inspect_run)
        target = next((run for run in runs if run.id == inspect_id), None)
        if target is None:
            print(f"Run id not found in sweep: {inspect_id}")
            return
        inspect_run_history(target, api=api, args=args, cache_dir=cache_dir, use_artifact=use_artifact, use_api=use_api)
        return

    scored: List[Dict[str, Any]] = []
    for run in runs:
        if args.state != "all" and run.state != args.state:
            continue

        rank_val, npts, source = score_run(
            run=run,
            api=api,
            args=args,
            cache_dir=cache_dir,
            use_artifact=use_artifact,
            use_api=use_api,
        )
        if rank_val is None:
            continue

        scored.append(
            {
                "rank": rank_val,
                "npts": npts,
                "name": run.name,
                "id": run.id,
                "state": run.state,
                "url": run.url,
                "source": source or "?",
                "run": run,
            }
        )

    if not scored:
        print("No runs found with usable history after burn-in.")
        return

    scored.sort(key=lambda x: x["rank"])
    topk = scored[: max(1, args.topk)]

    print(
        f"Top-{len(topk)} runs by mean(best {args.kbest}) {args.metric} after {args.step}>={int(args.burn)}"
    )
    print("-" * 80)
    for i, r in enumerate(topk, start=1):
        print(
            f"[{i}] rank={r['rank']:.6g}  npts={r['npts']:d}  src={r['source']}  state={r['state']} "
            f"name={r['name']}  id={r['id']}"
        )
        print(f"    {r['url']}")
    print("-" * 80)

    def is_hparam_key(k: str) -> bool:
        bad_prefixes = ("_", "wandb", "git", "slurm", "host", "cuda", "python")
        if k.startswith(bad_prefixes):
            return False
        return True

    print("Top-k configs (YAML-ish):")
    for i, r in enumerate(topk, start=1):
        cfg = {k: v for k, v in dict(r["run"].config).items() if is_hparam_key(str(k))}
        print(f"# rank {i}: {r['rank']:.6g}  run={r['name']}  id={r['id']}")
        for k, v in sorted(cfg.items()):
            print(f"{k}: {v}")
        print()


if __name__ == "__main__":
    main()
