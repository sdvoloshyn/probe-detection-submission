import argparse
import sys
from pathlib import Path
from typing import List

import yaml
import torch
from tqdm import tqdm


def resolve_device(dev_str: str) -> str:
    dev = str(dev_str).strip().lower()
    if dev.isdigit():
        return f"cuda:{dev}" if torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda"):
        return dev if torch.cuda.is_available() else "cpu"
    if dev == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if dev in {"cpu", "auto"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return "cpu"


def find_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference over a directory of images and save annotated outputs.")
    parser.add_argument("images_dir", type=str, help="Path to a directory containing images")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold, please run with 0.05 for inference")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    config_path = repo_root / "configs" / "final_config.yaml"
    weights_path = repo_root / "final.pt"
    output_dir = repo_root / "output"

    if not Path(args.images_dir).exists():
        print(f"❌ Images directory not found: {args.images_dir}")
        sys.exit(2)
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(2)
    if not weights_path.exists():
        print(f"❌ Model weights not found: {weights_path}")
        sys.exit(2)

    # Load minimal settings from config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    yolo_cfg = (cfg.get("yolo") or {})
    eval_cfg = (cfg.get("eval") or {})
    imgsz = int(yolo_cfg.get("imgsz", 640))
    iou = float(yolo_cfg.get("iou", 0.5))
    max_det = int(eval_cfg.get("max_det", 1))
    device_str = resolve_device(str(cfg.get("device", "cpu")))

    # Load YOLO model
    print("Initializing...")
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    model.to(device_str)
    # Force single-class name for plotting/labels
    try:
        model.names = {0: "probe"}
        if hasattr(model, "model") and hasattr(model.model, "names"):
            model.model.names = {0: "probe"}
    except Exception:
        pass

    # Collect images
    img_dir = Path(args.images_dir)
    images = find_images(img_dir)
    if not images:
        print(f"❌ No images found in: {img_dir}")
        sys.exit(2)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference and save annotated images (detections or 'no_pred')
    import cv2  # type: ignore

    # Match validation viz styling
    GREEN = (80, 220, 80)
    RED = (60, 60, 220)
    WHITE = (240, 240, 240)

    def _txt_bg(img, org, text, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, color=WHITE, thickness=1):
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = org
        pad = 2
        cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y + baseline), (0, 0, 0), -1)
        cv2.putText(img, text, (x + pad, y - pad), font, scale, color, thickness, cv2.LINE_AA)

    def draw_box(img, xyxy, color):
        H, W = img.shape[:2]
        x1, y1, x2, y2 = xyxy
        x1 = max(0, min(int(round(x1)), W))
        y1 = max(0, min(int(round(y1)), H))
        x2 = max(0, min(int(round(x2)), W))
        y2 = max(0, min(int(round(y2)), H))
        thickness = max(2, int(min(H, W) * 0.002))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    num_saved = 0
    for img_path in tqdm(images, desc="Inferencing", ncols=100):
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=float(args.conf),
            iou=iou,
            max_det=max_det,
            device=device_str,
            verbose=False,
            save=False,
            batch=1,
        )
        if not results:
            continue
        res = results[0]
        n_det = 0
        try:
            n_det = 0 if res.boxes is None else int(res.boxes.shape[0])
        except Exception:
            # Ultralytics results API: res.boxes is Boxes, length via len()
            try:
                n_det = len(res.boxes)  # type: ignore
            except Exception:
                n_det = 0

        if n_det > 0:
            # Post-filter to mirror validation behavior: conf, edge_touch_k, top max_det
            try:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
            except Exception:
                boxes_xyxy = []
                scores = []

            # Filter by confidence
            keep = [i for i, s in enumerate(scores) if float(s) >= float(args.conf)]

            # Keep top max_det by confidence
            if len(keep) > 0:
                keep = sorted(keep, key=lambda i: float(scores[i]), reverse=True)[:max_det]

            if len(keep) == 0:
                tqdm.write(f"No probe detected: {img_path.name}")
                # Save 'no_pred' annotated image
                img_np = cv2.imread(str(img_path))
                if img_np is not None:
                    _txt_bg(img_np, (10, 30), "no_pred")
                    out_path = output_dir / img_path.name
                    try:
                        cv2.imwrite(str(out_path), img_np)
                    except Exception as e:
                        print(f"Warning: failed to save {out_path}: {e}")
                continue

            # Draw manually to ensure consistent labeling
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                tqdm.write(f"Failed to load image for writing: {img_path.name}")
                continue
            for i in keep:
                x1, y1, x2, y2 = boxes_xyxy[i]
                draw_box(img_bgr, (x1, y1, x2, y2), RED)
                label = f"probe {float(scores[i]):.3f}"
                x1i, y1i = int(round(x1)), int(round(y1))
                _txt_bg(img_bgr, (max(0, x1i), max(20, y1i)), label)

            out_path = output_dir / img_path.name
            try:
                cv2.imwrite(str(out_path), img_bgr)
                num_saved += 1
            except Exception as e:
                print(f"Warning: failed to save {out_path}: {e}")
        else:
            tqdm.write(f"No probe detected: {img_path.name}")
            # Save 'no_pred' annotated image
            img_np = cv2.imread(str(img_path))
            if img_np is not None:
                _txt_bg(img_np, (10, 30), "no_pred")
                out_path = output_dir / img_path.name
                try:
                    cv2.imwrite(str(out_path), img_np)
                except Exception as e:
                    print(f"Warning: failed to save {out_path}: {e}")

    print(f"✅ Done. Saved {num_saved} annotated image(s) to: {output_dir}")


if __name__ == "__main__":
    main()


