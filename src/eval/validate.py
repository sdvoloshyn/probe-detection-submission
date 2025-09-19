"""
Validation utilities: single-class (probe) IoU/PR/F1 metrics with TensorBoard logging.
"""
from typing import Tuple, Dict, Any, List
import json
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between boxes a [Na,4] and b [Nb,4] in xyxy format."""
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    # areas
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    # intersections
    lt = torch.max(a[:, None, :2], b[None, :, :2])  # [Na, Nb, 2]
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])  # [Na, Nb, 2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / (union + 1e-7)
    return iou


@torch.no_grad()
def _evaluate_at_threshold(model, dataloader, conf_thresh: float, iou_thresh: float, device) -> Tuple[float, float, float, float]:
    """Evaluate model at a specific confidence threshold."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_sum = 0.0
    iou_cnt = 0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        # Ground truth boxes (xyxy) per image
        gt_boxes_list = [t['boxes'].to(device) for t in targets]

        # Predictions
        preds = model(images)

        # Iterate each image
        for pred, gt_boxes in zip(preds, gt_boxes_list):
            # Filter predictions by confidence and single class
            if isinstance(pred, dict):
                # RCNN-style output; ensure top-k per image honoring max_det from config
                pred_boxes = pred['boxes'].to(device)
                pred_scores = pred['scores'].to(device)
                keep = pred_scores >= conf_thresh
                pred_boxes = pred_boxes[keep]
                try:
                    max_det = int(getattr(config.eval, 'max_det', 1))
                except Exception:
                    max_det = 1
                if pred_boxes.shape[0] > max_det:
                    # keep highest scores
                    topk = torch.topk(pred_scores[keep], k=max_det).indices
                    pred_boxes = pred_boxes[topk]
            elif isinstance(pred, (list, tuple)) and len(pred) > 0 and isinstance(pred[0], dict):
                # Some torchvision versions return list[dict] even for a single image forward
                pb = pred[0]
                pred_boxes = pb.get('boxes', torch.empty((0,4), device=device)).to(device)
                pred_scores = pb.get('scores', torch.empty((0,), device=device)).to(device)
                keep = pred_scores >= conf_thresh
                pred_boxes = pred_boxes[keep]
                try:
                    max_det = int(getattr(config.eval, 'max_det', 1))
                except Exception:
                    max_det = 1
                if pred_boxes.shape[0] > max_det:
                    topk = torch.topk(pred_scores[keep], k=max_det).indices
                    pred_boxes = pred_boxes[topk]
            else:
                # Fallback empty if unknown format
                pred_boxes = torch.empty((0, 4), device=device)

            if gt_boxes.numel() == 0 and pred_boxes.numel() == 0:
                continue

            if gt_boxes.numel() == 0:
                total_fp += pred_boxes.shape[0]
                continue

            if pred_boxes.numel() == 0:
                total_fn += gt_boxes.shape[0]
                continue

            ious = _box_iou_xyxy(pred_boxes, gt_boxes)  # [Np, Ng]

            # Greedy matching by IoU
            matched_gt = set()
            matched_pred = set()
            # sort pairs by IoU descending
            npairs = []
            ious_cpu = ious.detach().cpu()
            for i in range(ious_cpu.shape[0]):
                for j in range(ious_cpu.shape[1]):
                    npairs.append((float(ious_cpu[i, j]), i, j))
            npairs.sort(reverse=True, key=lambda x: x[0])

            for iou, i, j in npairs:
                if iou < iou_thresh:
                    break
                if i in matched_pred or j in matched_gt:
                    continue
                matched_pred.add(i)
                matched_gt.add(j)
                total_tp += 1
                iou_sum += iou
                iou_cnt += 1

            total_fp += pred_boxes.shape[0] - len(matched_pred)
            total_fn += gt_boxes.shape[0] - len(matched_gt)

    precision = total_tp / (total_tp + total_fp + 1e-7)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    mean_iou = (iou_sum / max(iou_cnt, 1)) if iou_cnt > 0 else 0.0

    return float(precision), float(recall), float(f1), float(mean_iou)


@torch.no_grad()
def run_validation(model, dataloader, epoch: int, writer: SummaryWriter, config, save_dir: str = None) -> Dict[str, Any]:
    """
    Evaluate single-class detector on a dataloader with confidence sweep and log to TensorBoard.

    Args:
        model: torch.nn.Module (RCNN) or compatible detector in eval mode
        dataloader: iterable yielding (images, targets)
        epoch: current epoch index (int)
        writer: TensorBoard SummaryWriter
        config: global config with eval.iou_thresh and confidence thresholds
        save_dir: directory to save metrics JSON (optional)

    Returns:
        dict with best threshold results and full sweep data
    """
    device = next(model.parameters()).device
    model.eval()

    iou_thresh = config.eval.iou_thresh
    
    # Use confidence sweep from config, fallback to comprehensive default
    conf_thresholds = config.eval.conf_sweep
    
    best_f1 = -1.0
    best_threshold = conf_thresholds[0] if conf_thresholds else 0.5  # Use first threshold as fallback
    best_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mean_iou': 0.0}
    
    sweep_results = []
    
    print(f"Validation (epoch {epoch}): Running confidence sweep...")
    
    for conf_thresh in conf_thresholds:
        precision, recall, f1, mean_iou = _evaluate_at_threshold(
            model, dataloader, conf_thresh, iou_thresh, device
        )
        
        sweep_results.append({
            'threshold': conf_thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_iou': mean_iou
        })
        
        # Track best F1
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = conf_thresh
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mean_iou': mean_iou
            }
        
        print(f"  thresh={conf_thresh:.2f}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, IoU={mean_iou:.4f}")

    # Log best operating point metrics
    writer.add_scalar('val/best_conf_op', best_threshold, epoch)
    writer.add_scalar('val/f1_op', best_metrics['f1'], epoch)
    writer.add_scalar('val/precision_op', best_metrics['precision'], epoch)
    writer.add_scalar('val/recall_op', best_metrics['recall'], epoch)
    writer.add_scalar('val/mean_iou_tp_op', best_metrics['mean_iou'], epoch)
    
    # Log all thresholds for visualization
    # Do not spam per-threshold curves here; handled periodically elsewhere

    # Save metrics JSON if save_dir provided
    if save_dir:
        save_path = Path(save_dir) / f"metrics_epoch_{epoch}.json"
        metrics_data = {
            'epoch': epoch,
            'best_threshold': best_threshold,
            'best_metrics': best_metrics,
            'full_sweep': sweep_results
        }
        with open(save_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Generate validation visualizations if requested
        viz_enabled = config.eval.save_val_viz
        if viz_enabled:
            from .viz import save_val_viz_from_dataloader
            viz_dir = Path(save_dir) / "val_viz"
            save_val_viz_from_dataloader(model, dataloader, viz_dir, best_threshold, device)

    # Stdout print
    print(f"Validation (epoch {epoch}) - BEST: thresh={best_threshold:.2f}, P={best_metrics['precision']:.4f}, R={best_metrics['recall']:.4f}, F1={best_metrics['f1']:.4f}, mIoU={best_metrics['mean_iou']:.4f}")

    return {
        'best_threshold': best_threshold,
        'best_metrics': best_metrics,
        'full_sweep': sweep_results
    }


@torch.no_grad()
def quick_evalFixedConf(model,
                        dataset_yaml: str,
                        conf: float,
                        iou_thresh: float,
                        max_det: int,
                        edge_touch_k: int,
                        writer: SummaryWriter,
                        epoch: int,
                        cfg_eval,
                        out_dir: str,
                        total_epochs: int | None = None) -> Dict[str, Any]:
    """
    Lightweight per-epoch evaluation for YOLO models at fixed confidence.

    Runs Ultralytics model.val once with fixed conf and logs scalar summaries
    and optional histograms/visualizations to TensorBoard.
    """
    # Keep evaluation no-grad local; caller restores model.train() as needed
    import numpy as np
    from .metrics import mean_iou_tp_only, iou_percentiles_tp_only, Match
    from .op_select import build_val_cache
    from .viz import save_eval_viz_from_cache

    # Helper to canonicalize paths consistently
    from pathlib import Path as _P
    def _canon(p: str) -> str:
        return _P(p).resolve().as_posix()

    # Ultralytics validation to get standard metrics quickly
    results = None
    if hasattr(model, 'val'):
        # Run validation in eval mode
        was_training = model.training if hasattr(model, 'training') else False
        model.eval()
        results = model.val(
            data=dataset_yaml,
            split='val',
            conf=float(conf),
            iou=float(getattr(cfg_eval, 'nms_iou', getattr(cfg_eval, 'iou_thresh', 0.5))),
            max_det=int(max_det),
            save_json=True,
            save_hybrid=False,
            verbose=False
        )
        # Restore train mode on underlying module if it was training
        if was_training and hasattr(model, 'model') and hasattr(model.model, 'train'):
            model.model.train()

    # Fallback defaults
    map50 = float(results.box.map50) if results is not None else 0.0
    map5095 = float(results.box.map) if results is not None else 0.0

    # Build a minimal cache at this conf to compute TP-only IoUs and overlays
    # We reuse existing utilities to keep code unified
    try:
        # Infer labels from dataset.yaml path: images/val alongside labels/val
        import yaml
        with open(dataset_yaml, 'r') as f:
            ds = yaml.safe_load(f)
        ds_root = Path(ds['path'])
        val_images_dir = ds_root / ds['val']
        # Recover canonical labels via split files the trainer used earlier is not trivial here;
        # instead use the post-export labels next to images
        labels_dir = ds_root / 'labels' / 'val'
        # Build labels_dict from YOLO .txt files (single-class)
        labels_dict = {}
        val_paths: List[str] = []
        for img_file in sorted(Path(val_images_dir).glob('*.jpg')):
            key = _canon(str(img_file))
            val_paths.append(key)
            lab_file = labels_dir / (img_file.stem + '.txt')
            bboxes = []
            if lab_file.exists():
                try:
                    import numpy as np
                    H, W = 0, 0  # Not needed for xyxy conversion here; we exported val resized already
                    with open(lab_file, 'r') as lf:
                        for line in lf:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # YOLO normalized cx,cy,w,h -> approximate xyxy within [0,1] space
                                _, cx, cy, w, h = parts[:5]
                                cx = float(cx); cy = float(cy); w = float(w); h = float(h)
                                x1 = max(0.0, cx - w / 2.0)
                                y1 = max(0.0, cy - h / 2.0)
                                x2 = min(1.0, cx + w / 2.0)
                                y2 = min(1.0, cy + h / 2.0)
                                bboxes.append({'xyxy': [x1, y1, x2, y2]})
                except Exception:
                    pass
            labels_dict[key] = {'bboxes': bboxes}

        # Build cache at low base_conf and then filter to top-1 at provided conf
        pred_cache_all = build_val_cache(model=model,
                                         val_paths=val_paths,
                                         labels_dict=labels_dict,
                                         imgsz=getattr(getattr(cfg_eval, 'yolo', None), 'imgsz', 640) if hasattr(cfg_eval, 'yolo') else 640,
                                         base_conf=min(0.01, float(conf)),
                                         nms_iou=float(getattr(cfg_eval, 'iou_thresh', 0.5)))

        # Construct matches and metrics
        tp_matches: List[Match] = []
        total_tp = total_fp = total_fn = 0
        confs_top1 = []
        ious_tp = []
        from PIL import Image
        for img_path, dets in pred_cache_all.items():
            key = _canon(img_path)
            dets = [d for d in dets if d.conf >= float(conf)]
            # Load image dims (needed to normalize det boxes)
            im = Image.open(img_path)
            W, H = im.size
            # Edge filter
            if edge_touch_k and edge_touch_k > 0:
                dets = [d for d in dets if d.touches_edge(W, H, edge_touch_k)]
            # top-1
            dets = sorted(dets, key=lambda d: d.conf, reverse=True)[:int(max_det)]
            confs_top1.extend([d.conf for d in dets])

            gt = [b['xyxy'] for b in labels_dict[key]['bboxes']]
            if not gt and not dets:
                continue
            if not gt:
                total_fp += len(dets)
                continue
            if not dets:
                total_fn += len(gt)
                continue

            # Greedy one-to-one assignment by IoU in normalized space
            used = set()
            for det in dets:
                # Normalize detection to [0,1] using image size to match GT format
                dx1 = det.x1 / float(W)
                dy1 = det.y1 / float(H)
                dx2 = det.x2 / float(W)
                dy2 = det.y2 / float(H)
                best_iou = 0.0
                best_j = -1
                for j, g in enumerate(gt):
                    if j in used:
                        continue
                    # IoU in normalized coordinates
                    x1 = max(dx1, g[0]); y1 = max(dy1, g[1])
                    x2 = min(dx2, g[2]); y2 = min(dy2, g[3])
                    iw = max(0.0, x2 - x1)
                    ih = max(0.0, y2 - y1)
                    inter = iw * ih
                    det_a = (dx2 - dx1) * (dy2 - dy1)
                    gt_a = (g[2] - g[0]) * (g[3] - g[1])
                    union = det_a + gt_a - inter + 1e-9
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= float(iou_thresh) and best_j >= 0:
                    used.add(best_j)
                    total_tp += 1
                    ious_tp.append(best_iou)
                    tp_matches.append(Match(best_j, 0, best_iou, det.conf, img_path))
                else:
                    total_fp += 1
            total_fn += max(0, len(gt) - len(used))

        precision = total_tp / max(total_tp + total_fp, 1e-8)
        recall = total_tp / max(total_tp + total_fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        mean_iou_tp = float(np.mean(ious_tp)) if ious_tp else 0.0
        perc = iou_percentiles_tp_only(tp_matches, tuple(getattr(cfg_eval, 'track_percentiles', [0.5, 0.9])))

        # Optional negatives metrics
        neg_fp_rate = None
        pos_miss_rate = None
        num_neg = sum(1 for p in val_paths if len(labels_dict[p]['bboxes']) == 0)
        num_pos = len(val_paths) - num_neg
        if num_neg > 0:
            fp_on_neg = 0
            for img_path, dets in pred_cache_all.items():
                if len(labels_dict[img_path]['bboxes']) == 0:
                    dets_f = [d for d in dets if d.conf >= float(conf)]
                    fp_on_neg += 1 if len(dets_f) > 0 else 0
            neg_fp_rate = fp_on_neg / num_neg
        if num_pos > 0:
            miss_on_pos = 0
            for img_path, dets in pred_cache_all.items():
                if len(labels_dict[img_path]['bboxes']) > 0:
                    dets_f = [d for d in dets if d.conf >= float(conf)]
                    if len(dets_f) == 0:
                        miss_on_pos += 1
            pos_miss_rate = miss_on_pos / num_pos

        # Core OP metrics (rename with _op suffix)
        writer.add_scalar('val_epoch/precision_op', precision, epoch)
        writer.add_scalar('val_epoch/recall_op', recall, epoch)
        writer.add_scalar('val_epoch/f1_op', f1, epoch)
        writer.add_scalar('val_epoch/mean_iou_tp_op', mean_iou_tp, epoch)
        if neg_fp_rate is not None:
            writer.add_scalar('val_epoch/neg_fpr_op', float(neg_fp_rate), epoch)
        # Optional nice-to-haves
        if getattr(cfg_eval, 'log_conf_hist', False) and len(confs_top1) > 0:
            writer.add_histogram('val_epoch/conf_top1_op', np.asarray(confs_top1, dtype=np.float32), epoch)
        # Only log p50 and p90 to reduce clutter
        qkeys = {50:0.5, 90:0.9}
        for pct, q in qkeys.items():
            if q in perc:
                writer.add_scalar(f'val_epoch/iou_p{pct}_op', perc[q], epoch)

        # Histograms
        if getattr(cfg_eval, 'log_conf_hist', False) and len(confs_top1) > 0:
            writer.add_histogram('val_epoch/conf_top1', np.asarray(confs_top1, dtype=np.float32), epoch)
        if getattr(cfg_eval, 'log_iou_hist', False) and len(ious_tp) > 0:
            writer.add_histogram('val_epoch/iou_tp', np.asarray(ious_tp, dtype=np.float32), epoch)

        # Visualizations: mid-training (epoch ~ middle) and final only
        should_viz = False
        if getattr(cfg_eval, 'save_val_viz', False):
            if total_epochs and (epoch == total_epochs or epoch == max(1, total_epochs // 2)):
                should_viz = True
        if should_viz or int(getattr(cfg_eval, 'log_topk_worst', 0)) > 0:
            viz_dir = Path(out_dir) / f"val_viz/epoch_{epoch:03d}"
            # Save overlays from cache (top-1 policy inside viz)
            save_eval_viz_from_cache(pred_cache_all, labels_dict, str(viz_dir), class_name="probe")

            # Worst-K by IoU among TPs
            k = int(getattr(cfg_eval, 'log_topk_worst', 0) or 0)
            if k > 0 and tp_matches:
                tp_sorted = sorted(tp_matches, key=lambda m: m.iou if m.iou is not None else 1.0)[:k]
                worst_dir = viz_dir / 'worst_k'
                worst_dir.mkdir(parents=True, exist_ok=True)
                # Copy already-saved overlays if available
                try:
                    import shutil
                    for m in tp_sorted:
                        img_path = Path(m.img_path)
                        # find saved overlay path based on our viz naming convention
                        # We saved per subdir/image.jpg → replicate
                        src = viz_dir / img_path.parent.name / (img_path.stem + '.jpg')
                        if src.exists():
                            shutil.copy2(src, worst_dir / src.name)
                except Exception:
                    pass

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'mean_iou_tp': float(mean_iou_tp),
            'iou_percentiles': {str(k): float(v) for k, v in perc.items()},
            'neg_fp_rate': None if neg_fp_rate is None else float(neg_fp_rate),
            'pos_miss_rate': None if pos_miss_rate is None else float(pos_miss_rate)
        }
    except Exception as e:
        # Fail silently to avoid impacting training loop
        print(f"⚠️ quick_evalFixedConf failed: {e}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'map50': map50,
            'map50_95': map5095,
            'mean_iou_tp': 0.0,
            'iou_percentiles': {str(int(q*100)): 0.0 for q in getattr(cfg_eval, 'track_percentiles', [0.5, 0.75, 0.9])},
            'neg_fp_rate': None,
            'pos_miss_rate': None
        }


