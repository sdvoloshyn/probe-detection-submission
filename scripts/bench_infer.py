import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path so we can import the `src` package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import load_config  # noqa: E402
from src.config import Config as _Config  # noqa: E402
from src.config import DataConfig as _DataConfig  # noqa: E402
from src.config import AugmentationConfig as _AugConfig  # noqa: E402
from src.config import YOLOConfig as _YOLOConfig  # noqa: E402
from src.config import RCNNConfig as _RCNNConfig  # noqa: E402
from src.config import EvalConfig as _EvalConfig  # noqa: E402


# Track which FLOPs backend/convention was used during computation
FLOPS_BACKEND = None  # "fvcore" or "thop" (or None if failed)
FLOPS_CONVENTION = None  # "FLOPs (2 per MAC)" or "MACs (1 per MAC)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference speed for probe-detection models")
    parser.add_argument("--data", type=str, required=True, help="Path to folder with images")
    parser.add_argument("--weights", type=str, required=False, default=None,
                        help="Path to weights (.pt/.pth). If not provided, uses config or defaults")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string, e.g., cuda:0 or cpu")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16",
                        help="Precision for inference")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=300, help="Number of timed iterations")
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def set_runtime_flags() -> None:
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)


def list_images(folder: str) -> List[Path]:
    p = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    imgs = [f for f in sorted(p.iterdir()) if f.suffix.lower() in exts]
    return imgs


def load_image_cv2(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        return img  # BGR uint8
    except Exception:
        return None


def letterbox(image_bgr: np.ndarray, new_size: int, color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    import cv2
    h, w = image_bgr.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=resized.dtype)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess_for_yolo(image_bgr: np.ndarray, imgsz: int) -> torch.Tensor:
    # Letterbox to square, BGR->RGB, HWC->CHW, [0,1]
    img = letterbox(image_bgr, imgsz)
    img = img[:, :, ::-1]  # BGR->RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # CHW
    return img.unsqueeze(0)  # 1xCxHxW


def preprocess_for_rcnn(image_bgr: np.ndarray, imgsz: int) -> List[torch.Tensor]:
    # Simple resize to square, convert to RGB float [0,1]
    import cv2
    img = cv2.resize(image_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1]  # BGR->RGB
    img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    return [img]


def get_commit_sha() -> str:
    try:
        import subprocess
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)).decode().strip()
        return sha
    except Exception:
        return "N/A"


def compute_params_millions(model: nn.Module) -> float:
    try:
        total_params = sum(p.numel() for p in model.parameters())
        return round(total_params / 1e6, 3)
    except Exception:
        return float("nan")


def compute_flops_g(model: nn.Module, sample_input: Any) -> Optional[float]:
    global FLOPS_BACKEND, FLOPS_CONVENTION
    # Try fvcore first (handles list inputs better); then thop
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
        model.eval()
        with torch.no_grad():
            flops = FlopCountAnalysis(model, sample_input).total()
        # fvcore counts FLOPs where multiply and add are two separate operations (2 per MAC)
        FLOPS_BACKEND = "fvcore"
        FLOPS_CONVENTION = "FLOPs (2 per MAC)"
        return round(flops / 1e9, 3)
    except Exception:
        pass
    try:
        from thop import profile  # type: ignore
        model.eval()
        with torch.no_grad():
            if isinstance(sample_input, (list, tuple)):
                flops, _ = profile(model, inputs=tuple(sample_input))
            else:
                flops, _ = profile(model, inputs=(sample_input,))
        # THOP commonly reports MACs by default (multiply–add counted as one)
        FLOPS_BACKEND = "thop"
        FLOPS_CONVENTION = "MACs (1 per MAC)"
        return round(flops / 1e9, 3)
    except Exception:
        return None


def tolerant_load_config(config_path: str) -> _Config:
    """Load YAML config but ignore unknown keys for nested dataclasses.

    This mirrors src.config.load_config but filters extra keys that might have
    been introduced in some configs (e.g., nms_iou_sweep in eval).
    """
    path = Path(config_path)
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    def filter_fields(src: dict, cls) -> dict:
        allowed = set(getattr(cls, "__dataclass_fields__").keys())
        return {k: v for k, v in (src or {}).items() if k in allowed}

    data = _DataConfig(**filter_fields(cfg_dict.get("data", {}), _DataConfig))
    aug = _AugConfig(**filter_fields(cfg_dict.get("aug", {}), _AugConfig))
    yolo = _YOLOConfig(**filter_fields(cfg_dict.get("yolo", {}), _YOLOConfig))
    rcnn = _RCNNConfig(**filter_fields(cfg_dict.get("rcnn", {}), _RCNNConfig))
    eval_cfg = _EvalConfig(**filter_fields(cfg_dict.get("eval", {}), _EvalConfig))

    cfg = _Config(
        seed=cfg_dict.get("seed", 42),
        device=cfg_dict.get("device", "cpu"),
        model=cfg_dict.get("model", "yolo_nano"),
        data=data,
        aug=aug,
        yolo=yolo,
        rcnn=rcnn,
        eval=eval_cfg,
        name=cfg_dict.get("name"),
        save_dir=cfg_dict.get("save_dir"),
    )
    return cfg


def load_yolo_model(model_type: str, weights_path: Optional[str], device: torch.device) -> Tuple[Any, nn.Module]:
    from ultralytics import YOLO  # lazy import
    def _infer_default_weights_from_model(name: str) -> str:
        name = str(name).lower().strip()
        if name in {"yolo_nano"}:
            return "yolov8n.pt"
        import re
        m = re.match(r"yolo[_\-]?v?(\d+)(?:[_\-]?(nano|n|s|m|l|x))?$", name)
        if not m:
            return "yolov8n.pt"
        ver, size = m.group(1), (m.group(2) or "s")
        size = {"nano": "n"}.get(size, size)
        prefix = "yolov" if ver == "8" else "yolo"
        return f"{prefix}{ver}{size}.pt"
    if not weights_path:
        weights_path = _infer_default_weights_from_model(model_type)
    yolo = YOLO(weights_path)
    yolo.to(str(device))
    backbone = yolo.model  # nn.Module used for FLOPs/params
    backbone.eval()
    return yolo, backbone


def load_rcnn_model(config, weights_path: Optional[str], device: torch.device) -> nn.Module:
    # Reuse trainer's initializer to build proper architecture
    from src.trainers.rcnn_trainer import RCNNTrainer
    trainer = RCNNTrainer(config)
    model = trainer._initialize_model()
    model.to(device)
    if weights_path and weights_path not in {"pretrained", "DEFAULT"} and Path(weights_path).exists():
        sd = torch.load(weights_path, map_location=device)
        # Accept either full state dict or nested under 'model' key
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            # Some checkpoints may have keys with module. prefix
            new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def benchmark(
    model_type: str,
    model_runner: Any,
    backbone_for_stats: nn.Module,
    device: torch.device,
    precision: str,
    warmup_iters: int,
    timed_iters: int,
    preprocessed_inputs: List[Any],
) -> Dict[str, Any]:
    use_cuda = device.type == "cuda"
    use_fp16 = (precision == "fp16") and use_cuda

    # Warmup with dummy data to settle kernels
    imgsz = preprocessed_inputs[0][0].shape[-1] if model_type.startswith("yolo") else None
    if model_type.startswith("yolo"):
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=torch.float32)
        dummy_in = dummy
    else:
        # RCNN expects list of tensors
        dummy = torch.zeros(3, preprocessed_inputs[0][0].shape[-2], preprocessed_inputs[0][0].shape[-1],
                            device=device, dtype=torch.float32)
        dummy_in = [dummy]

    for _ in range(max(0, warmup_iters)):
        if use_cuda:
            torch.cuda.synchronize()
        with torch.cuda.amp.autocast(enabled=use_fp16):
            _ = model_runner(dummy_in)
        if use_cuda:
            torch.cuda.synchronize()

    # Reset peak memory tracking
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms: List[float] = []
    total_images = 0
    pbar = tqdm(total=timed_iters, desc="Benchmarking", ncols=100, leave=False)
    for i in range(timed_iters):
        inp = preprocessed_inputs[i % len(preprocessed_inputs)]
        # Ensure input is on device and correct dtype
        if model_type.startswith("yolo"):
            x = inp[0].to(device, non_blocking=True)
            feed = x
        else:
            x = inp[0].to(device, non_blocking=True)
            feed = [x]

        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=use_fp16):
            _ = model_runner(feed)
        if use_cuda:
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        total_images += 1
        pbar.update(1)
    pbar.close()

    peak_mem_mb = 0.0
    if use_cuda:
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)

    # Stats
    lat_np = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(lat_np.mean()) if lat_np.size else float("nan")
    p50 = float(np.percentile(lat_np, 50)) if lat_np.size else float("nan")
    p90 = float(np.percentile(lat_np, 90)) if lat_np.size else float("nan")
    p95 = float(np.percentile(lat_np, 95)) if lat_np.size else float("nan")
    fps = 1000.0 / mean_ms if mean_ms and mean_ms > 0 else float("nan")

    # Params and FLOPs
    params_m = compute_params_millions(backbone_for_stats)
    sample_for_flops: Any
    if model_type.startswith("yolo"):
        sample_for_flops = torch.zeros(1, 3, preprocessed_inputs[0][0].shape[-2], preprocessed_inputs[0][0].shape[-1])
    else:
        sample_for_flops = [torch.zeros(3, preprocessed_inputs[0][0].shape[-2], preprocessed_inputs[0][0].shape[-1])]
    try:
        flops_g = compute_flops_g(backbone_for_stats.cpu(), sample_for_flops)
    except Exception:
        flops_g = None
    finally:
        backbone_for_stats.to(device)

    return {
        "latencies_ms": latencies_ms,
        "mean_ms": round(mean_ms, 3) if not np.isnan(mean_ms) else float("nan"),
        "p50_ms": round(p50, 3) if not np.isnan(p50) else float("nan"),
        "p90_ms": round(p90, 3) if not np.isnan(p90) else float("nan"),
        "p95_ms": round(p95, 3) if not np.isnan(p95) else float("nan"),
        "fps": round(fps, 3) if not np.isnan(fps) else float("nan"),
        "params_M": params_m,
        "flops_G": flops_g if flops_g is not None else "N/A",
        "peak_mem_MB": round(peak_mem_mb, 3),
        "total_images": total_images,
    }


def main() -> None:
    args = parse_args()
    set_runtime_flags()

    device = resolve_device(args.device)

    # Load config for model selection and imgsz
    # Try strict loader first, fallback to tolerant if extra keys present
    try:
        cfg = load_config(args.config)
    except TypeError:
        cfg = tolerant_load_config(args.config)
    model_type = cfg.model
    if not (str(model_type).startswith("yolo") or model_type == "faster_rcnn"):
        raise ValueError(f"Unknown model type in config: {model_type}")

    imgsz = cfg.yolo.imgsz if str(model_type).startswith("yolo") else cfg.rcnn.imgsz

    # Resolve weights
    weights_path = args.weights
    if weights_path is None:
        if str(model_type).startswith("yolo"):
            weights_path = cfg.yolo.weights
        else:
            weights_path = cfg.rcnn.weights

    # Load model
    if str(model_type).startswith("yolo"):
        yolo_model, backbone = load_yolo_model(model_type, weights_path, device)

        def runner(feed):
            return yolo_model(feed)

        preprocess = lambda img: preprocess_for_yolo(img, imgsz)
    else:
        rcnn_model = load_rcnn_model(cfg, weights_path, device)
        backbone = rcnn_model

        def runner(feed):
            return rcnn_model(feed)

        preprocess = lambda img: preprocess_for_rcnn(img, imgsz)

    # Collect and preprocess images (exclude I/O from timing)
    img_paths = list_images(args.data)
    if len(img_paths) == 0:
        raise FileNotFoundError(f"No images found in: {args.data}")

    preprocessed: List[Any] = []
    skipped = 0
    for p in img_paths:
        arr = load_image_cv2(p)
        if arr is None:
            skipped += 1
            continue
        try:
            x = preprocess(arr)
            preprocessed.append((x,))
        except Exception:
            skipped += 1
            continue

    if len(preprocessed) == 0:
        raise RuntimeError("All images failed to load/preprocess.")

    # Run benchmark
    results = benchmark(
        model_type=model_type,
        model_runner=runner,
        backbone_for_stats=backbone,
        device=device,
        precision=args.precision,
        warmup_iters=args.warmup,
        timed_iters=args.iters,
        preprocessed_inputs=preprocessed,
    )

    # Commit/metadata
    commit_sha = get_commit_sha()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    precision = args.precision
    device_str = str(device)

    # Console print
    print("\n==== Inference Benchmark ====")
    print(f"Date: {date_str}")
    print(f"Model: {model_type}")
    print(f"Image size: {imgsz}")
    print(f"Precision: {precision}")
    print(f"Device: {device_str}")
    print(f"Params (M): {results['params_M']}")
    print(f"FLOPs (G): {results['flops_G']}")
    print(f"Mean latency (ms): {results['mean_ms']}")
    print(f"p50 (ms): {results['p50_ms']}")
    print(f"p90 (ms): {results['p90_ms']}")
    print(f"p95 (ms): {results['p95_ms']}")
    print(f"FPS: {results['fps']}")
    print(f"Peak GPU mem (MB): {results['peak_mem_MB']}")
    print(f"Total images processed: {results['total_images']}")
    print(f"Commit: {commit_sha}")
    # Clarify which FLOPs backend and counting convention were used
    try:
        backend = FLOPS_BACKEND or "N/A"
        convention = FLOPS_CONVENTION or "N/A"
        print(f"FLOPs backend: {backend} — {convention}")
    except Exception:
        pass

    # CSV append
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmarks.csv"
    row = [
        date_str,
        model_type,
        imgsz,
        precision,
        device_str,
        results["params_M"],
        results["flops_G"],
        results["mean_ms"],
        results["p50_ms"],
        results["p90_ms"],
        results["p95_ms"],
        results["fps"],
        results["peak_mem_MB"],
        commit_sha,
    ]
    header = [
        "date",
        "model_type",
        "imgsz",
        "precision",
        "device",
        "params_M",
        "flops_G",
        "mean_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "fps",
        "peak_mem_MB",
        "commit_sha",
    ]
    write_header = not csv_path.exists()
    try:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        print(f"Warning: failed to append CSV at {csv_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()


