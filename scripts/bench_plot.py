import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from scripts.reporting import RUN_NAME_TO_VIZ_NAME  # type: ignore
except Exception:
    RUN_NAME_TO_VIZ_NAME = {}

from src.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot F1@best vs inference throughput (FPS) for selected runs")
    p.add_argument("--runs", nargs="*", default=None, help="Run names to include; order controls labeling")
    p.add_argument("--results_dir", default=str(ROOT / "results"), help="Directory with results_*.xlsx")
    p.add_argument("--runs_root", default=str(ROOT / "runs"), help="Root directory containing run folders")
    p.add_argument("--device", default="cuda:0", help="Device for benchmarking (e.g., cuda:0 or cpu)")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16", help="AMP precision for timing")
    p.add_argument("--warmup", type=int, default=20, help="Warmup iterations before timing")
    p.add_argument("--iters", type=int, default=200, help="Timed iterations per run")
    p.add_argument("--data", type=str, required=True, help="Folder with images to benchmark on")
    p.add_argument("--save", type=str, default=str(ROOT / "results" / "bench_plot_f1_vs_time.png"), help="Path to save the plot PNG")
    return p.parse_args()


def load_compact_results(results_dir: Path) -> pd.DataFrame:
    xlsx = results_dir / "results_compact.xlsx"
    if not xlsx.exists():
        raise FileNotFoundError(f"Missing {xlsx}")
    df = pd.read_excel(xlsx, sheet_name="results", header=[0, 1])
    return df


def load_extended_results(results_dir: Path) -> Optional[pd.DataFrame]:
    xlsx = results_dir / "results_extended.xlsx"
    if not xlsx.exists():
        return None
    try:
        df = pd.read_excel(xlsx, sheet_name=0, header=[0, 1])
        return df
    except Exception:
        return None


def get_runs_in_order(df_compact: pd.DataFrame, requested_runs: Optional[List[str]]) -> List[str]:
    top = "run_name"
    if top not in df_compact.columns.get_level_values(0):
        raise ValueError("results_compact.xlsx missing top-level 'run_name' column")
    sub = df_compact[top].columns[0]
    col = (top, sub)
    if not requested_runs:
        return list(df_compact[col].astype(str).unique())
    return list(requested_runs)


def extract_mean_f1_for_runs(df_compact: pd.DataFrame, runs: List[str]) -> Dict[str, float]:
    if "F1@best" not in df_compact.columns.get_level_values(0):
        raise ValueError("'F1@best' not found in results_compact.xlsx")
    top = "run_name"
    sub = df_compact[top].columns[0]
    run_col = (top, sub)
    df_idx = df_compact.set_index(run_col)
    out: Dict[str, float] = {}
    for run in runs:
        if run not in df_idx.index:
            continue
        col = ("F1@best", "mean")
        try:
            out[run] = float(df_idx.loc[run][col])
        except Exception:
            out[run] = float("nan")
    return out


def find_best_weights_for_run(run_name: str, extended: Optional[pd.DataFrame], runs_root: Path) -> Optional[Path]:
    # Prefer extended results' stored best path
    if extended is not None:
        try:
            top = "Aggregates"; sub = "weights_best_path"
            run_top = "Run identity"; run_sub = "run_name"
            if top in extended.columns.get_level_values(0) and run_top in extended.columns.get_level_values(0):
                run_col = (run_top, run_sub)
                df_idx = extended.set_index(run_col)
                if run_name in df_idx.index and (top, sub) in extended.columns:
                    p = Path(str(df_idx.loc[run_name][(top, sub)]))
                    if p.exists():
                        return p
        except Exception:
            pass

    # Fallback: search this run's folder, supporting fold_* layouts
    candidate = None
    try:
        run_dir = runs_root / run_name
        bases = [run_dir] if run_dir.exists() else [runs_root]
        patterns = [
            "**/weights/best_es.pt", "**/weights/best.pt", "**/weights/last.pt",
            "**/best_model.pth", "**/final_model.pth",
        ]
        for base in bases:
            for pat in patterns:
                for p in base.rglob(pat):
                    parts = list(p.parents)
                    rn = parts[2].name if len(parts) > 2 else ""
                    if (base == run_dir and run_dir in p.parents) or (rn == run_name):
                        candidate = p
                        break
                if candidate is not None:
                    break
            if candidate is not None:
                break
    except Exception:
        candidate = None
    return candidate


def load_fold_config_from_weights(weights_path: Path) -> Optional[Path]:
    run_dir = weights_path.parent.parent
    for name in ["config.yaml", "config.json"]:
        p = run_dir / name
        if p.exists():
            return p
    return None


def resolve_model_and_imgsz(config_path: Path) -> Tuple[str, int]:
    if config_path.suffix.lower() == ".json":
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        from src.config import Config as _Config, DataConfig as _Data, AugmentationConfig as _Aug, YOLOConfig as _YOLO, RCNNConfig as _RCNN, EvalConfig as _Eval
        def get(d, k, default=None):
            return d.get(k, default) if isinstance(d, dict) else default
        data = _Data(**get(cfg_dict, "data", {}))
        aug = _Aug(**get(cfg_dict, "aug", {}))
        yolo = _YOLO(**get(cfg_dict, "yolo", {}))
        rcnn = _RCNN(**get(cfg_dict, "rcnn", {}))
        eval_cfg = _Eval(**get(cfg_dict, "eval", {}))
        cfg = _Config(seed=get(cfg_dict, "seed", 42), device=get(cfg_dict, "device", "cpu"), model=get(cfg_dict, "model", "yolo_nano"), data=data, aug=aug, yolo=yolo, rcnn=rcnn, eval=eval_cfg, name=get(cfg_dict, "name"), save_dir=get(cfg_dict, "save_dir"))
    else:
        cfg = load_config(str(config_path))
    model_type = cfg.model
    imgsz = cfg.yolo.imgsz if str(model_type).startswith("yolo") else cfg.rcnn.imgsz
    return str(model_type), int(imgsz)


def compute_inference_time_ms(model_type: str, weights_path: Path, imgsz: int, device: str, precision: str, warmup: int, iters: int, data_dir: str) -> float:
    import torch
    from scripts.bench_infer import list_images, load_image_cv2, preprocess_for_yolo, preprocess_for_rcnn, resolve_device, load_yolo_model, load_rcnn_model, benchmark, tolerant_load_config, set_runtime_flags  # type: ignore

    try:
        set_runtime_flags()
    except Exception:
        pass

    device_obj = resolve_device(device)

    cfg = None
    if not str(model_type).startswith("yolo"):
        cfg_path = load_fold_config_from_weights(weights_path)
        cfg = tolerant_load_config(str(cfg_path)) if cfg_path is not None else None

    if str(model_type).startswith("yolo"):
        yolo_model, backbone = load_yolo_model(model_type, str(weights_path), device_obj)
        runner = lambda feed: yolo_model(feed)
        preprocess = lambda img: preprocess_for_yolo(img, imgsz)
    else:
        if cfg is None:
            raise RuntimeError(f"Cannot load RCNN config near {weights_path}")
        rcnn_model = load_rcnn_model(cfg, str(weights_path), device_obj)
        backbone = rcnn_model
        runner = lambda feed: rcnn_model(feed)
        preprocess = lambda img: preprocess_for_rcnn(img, imgsz)

    img_paths = list_images(data_dir)
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {data_dir}")
    preprocessed: List[Tuple] = []
    for p in img_paths[:max(1, min(len(img_paths), 32))]:
        arr = load_image_cv2(p)
        if arr is None:
            continue
        try:
            x = preprocess(arr)
            preprocessed.append((x,))
        except Exception:
            continue
    if not preprocessed:
        raise RuntimeError("Failed to preprocess images for benchmarking")

    stats = benchmark(
        model_type=str(model_type),
        model_runner=runner,
        backbone_for_stats=backbone,
        device=device_obj,
        precision=precision,
        warmup_iters=warmup,
        timed_iters=iters,
        preprocessed_inputs=preprocessed,
    )
    return float(stats.get("mean_ms", float("nan")))


def plot_f1_vs_time(points: List[Tuple[float, float, str]], save_path: Path) -> None:
    if not points:
        raise ValueError("No points to plot")
    points = [(f, t, l) for (f, t, l) in points if not (np.isnan(f) or np.isnan(t))]
    if not points:
        raise ValueError("All points are NaN")

    fps_points: List[Tuple[float, float, str]] = []
    for f1, ms, lab in points:
        fps = 1000.0 / ms if (ms and ms > 0) else float("nan")
        fps_points.append((f1, fps, lab))

    fps_points.sort(key=lambda x: (x[0], x[1]), reverse=True)
    f1_vals = [p[0] for p in fps_points]
    fps_vals = [p[1] for p in fps_points]
    labels = [p[2] for p in fps_points]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(fps_vals, f1_vals, s=64, edgecolor="black", linewidth=0.6, alpha=0.9)
    for x, y, lab in zip(fps_vals, f1_vals, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6), ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.8))
    ax.set_xlabel("Throughput (FPS)")
    ax.set_ylabel("Mean F1@best")
    ax.set_title("F1@best vs Inference Throughput")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    runs_root = Path(args.runs_root)
    df_compact = load_compact_results(results_dir)
    df_extended = load_extended_results(results_dir)
    runs = get_runs_in_order(df_compact, args.runs)

    mean_f1 = extract_mean_f1_for_runs(df_compact, runs)

    points: List[Tuple[float, float, str]] = []
    for run in runs:
        f1 = float(mean_f1.get(run, float("nan")))
        weights = find_best_weights_for_run(run, df_extended, runs_root)
        if weights is None:
            print(f"Warning: could not find best weights for run '{run}'")
            continue
        cfg_path = load_fold_config_from_weights(weights)
        if cfg_path is None:
            print(f"Warning: missing config near weights for run '{run}' at {weights}")
            continue
        model_type, imgsz = resolve_model_and_imgsz(cfg_path)
        # Debug: show exactly what we are benchmarking per run
        try:
            meta_model = None
            if cfg_path.suffix == ".json":
                with open(cfg_path, "r") as f:
                    meta_model = json.load(f).get("model")
            else:
                import yaml as _yaml
                with open(cfg_path, "r") as f:
                    meta_model = _yaml.safe_load(f).get("model")
        except Exception:
            meta_model = None
        print(f"[bench_plot] Run='{run}' -> weights='{weights}', model='{meta_model or model_type}', imgsz={imgsz}")

        try:
            mean_ms = compute_inference_time_ms(
                model_type=model_type,
                weights_path=weights,
                imgsz=imgsz,
                device=args.device,
                precision=args.precision,
                warmup=args.warmup,
                iters=args.iters,
                data_dir=args.data,
            )
        except Exception as e:
            print(f"Warning: benchmark failed for run '{run}': {e}")
            continue
        label = RUN_NAME_TO_VIZ_NAME.get(run, run)
        points.append((f1, mean_ms, label))

    if not points:
        raise RuntimeError("No valid points to plot")
    plot_f1_vs_time(points, Path(args.save))
    print(f"Saved plot to {args.save}")


if __name__ == "__main__":
    main()


