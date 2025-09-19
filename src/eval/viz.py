# src/eval/viz.py
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Any

GREEN = (80, 220, 80)
RED   = (60, 60, 220)
WHITE = (240, 240, 240)

def _txt_bg(img, org, text, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, color=WHITE, thickness=1):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    pad = 2
    cv2.rectangle(img, (x, y - th - 2*pad), (x + tw + 2*pad, y + baseline), (0,0,0), -1)
    cv2.putText(img, text, (x + pad, y - pad), font, scale, color, thickness, cv2.LINE_AA)

def _clamp_xyxy(xyxy, W, H):
    """Clamp bounding box coordinates to image bounds."""
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    return [x1, y1, x2, y2]

def draw_box(img, xyxy, color, thickness=2):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in _clamp_xyxy(xyxy, W, H)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def save_val_viz(samples, preds_by_path, out_dir, class_name="probe"):
    """
    samples: iterable of dicts with keys: image_path (str), bboxes (list of xyxy GTs)
    preds_by_path: dict[str] -> list of predictions [{xyxy:[...], score:float}] (can be empty)
    out_dir: Path-like output directory
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    for s in samples:
        img_path = s["image_path"]
        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]

        # Draw GT (green)
        for gt in s.get("bboxes", []):
            xyxy = gt["xyxy"] if isinstance(gt, dict) else gt
            draw_box(img, xyxy, GREEN, thickness=max(2, int(min(H,W) * 0.002)))

        # Draw top-1 pred (red + score)
        preds = preds_by_path.get(img_path, [])
        if preds:
            top = max(preds, key=lambda p: p.get("score", -1.0))
            draw_box(img, top["xyxy"], RED, thickness=max(2, int(min(H,W) * 0.002)))
            x1, y1, _, _ = [int(round(v)) for v in top["xyxy"]]
            _txt_bg(img, (max(0, x1), max(20, y1)), f"{class_name} {top.get('score', 0.0):.3f}")

        # If no preds, optionally annotate
        else:
            _txt_bg(img, (10, 30), "no_pred")

        rel = Path(img_path)
        subdir = outp / rel.parent.name
        subdir.mkdir(parents=True, exist_ok=True)
        out_file = subdir / (rel.stem + ".jpg")
        cv2.imwrite(str(out_file), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def save_eval_viz_from_cache(pred_cache: Dict[str, List], labels_dict: Dict[str, Any], 
                            out_dir: str, class_name: str = "probe"):
    """
    Generate validation visualizations from prediction cache and labels dict.
    
    Args:
        pred_cache: Dict mapping image_path -> List[Detection] (from op_select.py)
        labels_dict: Dict mapping image_path -> {'bboxes': [{'xyxy': [...]}]}
        out_dir: Output directory for visualization images
        class_name: Class name to display (default: "probe")
    """
    from .op_select import Detection
    
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating validation visualizations for {len(pred_cache)} images...")
    
    for img_path, detections in pred_cache.items():
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Warning: Could not load image {img_path}")
            continue

        H, W = img.shape[:2]

        # Draw GT (green)
        gt_data = labels_dict[img_path]
        for bbox_data in gt_data['bboxes']:
            xyxy = bbox_data['xyxy']
            draw_box(img, xyxy, GREEN, thickness=max(2, int(min(H,W) * 0.002)))

        # Draw top-1 pred (red + score)
        if detections:
            # Sort by confidence and take the highest
            top_det = max(detections, key=lambda d: d.conf)
            xyxy = [top_det.x1, top_det.y1, top_det.x2, top_det.y2]
            draw_box(img, xyxy, RED, thickness=max(2, int(min(H,W) * 0.002)))
            x1, y1, _, _ = [int(round(v)) for v in xyxy]
            _txt_bg(img, (max(0, x1), max(20, y1)), f"{class_name} {top_det.conf:.3f}")
        else:
            _txt_bg(img, (10, 30), "no_pred")

        # Save image
        rel = Path(img_path)
        subdir = outp / rel.parent.name
        subdir.mkdir(parents=True, exist_ok=True)
        out_file = subdir / (rel.stem + ".jpg")
        cv2.imwrite(str(out_file), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    
    print(f"✅ Saved validation visualizations to {outp}/")


def save_val_viz_from_dataloader(model, dataloader, out_dir: str, conf_thresh: float, 
                                device, class_name: str = "probe"):
    """
    Generate validation visualizations from a dataloader and model.
    
    Args:
        model: PyTorch model in eval mode
        dataloader: DataLoader yielding (images, targets)
        out_dir: Output directory for visualization images
        conf_thresh: Confidence threshold for predictions
        device: Device for model inference
        class_name: Class name to display (default: "probe")
    """
    import torch
    
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating validation visualizations from dataloader (conf={conf_thresh:.3f})...")
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(device) for img in images]
            
            # Get predictions
            preds = model(images)
            
            # Process each image in the batch
            for i, (pred, target) in enumerate(zip(preds, targets)):
                # Extract image path from target if available
                img_path = target.get('image_path', f'batch_{batch_idx}_img_{i}.jpg')
                
                # Load original image
                img = cv2.imread(img_path) if isinstance(img_path, str) else None
                if img is None:
                    print(f"  Warning: Could not load image {img_path}")
                    continue
                
                H, W = img.shape[:2]
                
                # Draw GT (green)
                if 'boxes' in target:
                    gt_boxes = target['boxes'].cpu().numpy()
                    for box in gt_boxes:
                        draw_box(img, box, GREEN, thickness=max(2, int(min(H,W) * 0.002)))
                
                # Draw top-1 pred (red + score)
                if isinstance(pred, dict) and 'boxes' in pred and 'scores' in pred:
                    pred_boxes = pred['boxes'].cpu().numpy()
                    pred_scores = pred['scores'].cpu().numpy()
                    
                    # Filter by confidence and take top prediction
                    valid_mask = pred_scores >= conf_thresh
                    if valid_mask.any():
                        valid_scores = pred_scores[valid_mask]
                        valid_boxes = pred_boxes[valid_mask]
                        top_idx = valid_scores.argmax()
                        top_box = valid_boxes[top_idx]
                        top_score = valid_scores[top_idx]
                        
                        draw_box(img, top_box, RED, thickness=max(2, int(min(H,W) * 0.002)))
                        x1, y1, _, _ = [int(round(v)) for v in top_box]
                        _txt_bg(img, (max(0, x1), max(20, y1)), f"{class_name} {top_score:.3f}")
                    else:
                        _txt_bg(img, (10, 30), "no_pred")
                else:
                    _txt_bg(img, (10, 30), "no_pred")
                
                # Save image
                rel = Path(img_path)
                subdir = outp / rel.parent.name
                subdir.mkdir(parents=True, exist_ok=True)
                out_file = subdir / (rel.stem + ".jpg")
                cv2.imwrite(str(out_file), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    
    print(f"✅ Saved validation visualizations to {outp}/")
