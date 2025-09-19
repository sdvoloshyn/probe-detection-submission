from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List
import json
import statistics
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from src.cv.lomo import LomoFold, load_lomo_folds_from_dir
from src.config import load_config
from src.main import train_model
from src.eval.sweep import select_operating_point
from src.datasets.canonical import load_canonical_dataset
from src.utils.run_logger import RunLogger


def run_lomo_cv(
    config_path: str,
    labels_json: str,
    splits_root: str,
    base_log_dir: str = "runs",
    run_timestamp: str | None = None,
) -> Dict:
    """
    Run Leave-One-Mission-Out CV across all missions.

    Returns dict with per-fold metrics and aggregate mean/std where available.
    """
    # Stable run identity
    cv_timestamp = run_timestamp or datetime.now().astimezone().isoformat()
    base_name = Path(base_log_dir).name
    # Write CV context so trainers can upsert into the same compact row
    try:
        ctx = {"run_name": base_name, "timestamp": cv_timestamp}
        with open(Path(base_log_dir) / "cv_context.json", "w") as f:
            json.dump(ctx, f)
    except Exception:
        pass

    # Load pre-created splits (do not recreate here)
    folds: List[LomoFold] = load_lomo_folds_from_dir(splits_root)
    if not folds:
        raise RuntimeError(
            "No LOMO splits found under '{}'\n"
            "Please create them once using:\n"
            "  python data/prepare_data_lomo_splits.py --labels-json {} --output {}".format(
                splits_root, labels_json, splits_root
            )
        )

    # Results store
    per_fold_metrics: List[Dict] = []

    # Iterate folds
    for fold in folds:
        fold_name = f"fold_{fold.fold_index}"
        fold_log_dir = Path(base_log_dir) / fold_name
        fold_log_dir.mkdir(parents=True, exist_ok=True)

        # For each fold, we need a config override pointing to fold's split files
        cfg = load_config(config_path)
        # Point to fold split files
        cfg.data.splits_dir = str(fold.fold_dir)

        # Create temporary config file for this fold
        fold_config_path = fold_log_dir / "config.yaml"
        import yaml
        with open(fold_config_path, "w") as f:
            yaml.dump(asdict(cfg), f, default_flow_style=False)

        # Persist effective fold config as JSON too
        with open(fold_log_dir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)

        # TensorBoard per-fold under dedicated subdir
        tb_dir = Path(fold_log_dir) / "tensorboard"
        writer = SummaryWriter(str(tb_dir))

        # Write cv_context.json into each fold dir so trainers upsert into the same compact row
        try:
            with open(Path(fold_log_dir) / 'cv_context.json', 'w') as f:
                json.dump({
                    'run_name': base_name,
                    'timestamp': cv_timestamp,
                    'fold_index': int(fold.fold_index)
                }, f)
        except Exception:
            pass

        # Train using the fold-specific config
        weights_path = train_model(str(fold_config_path), dry_run=False, writer=writer, run_dir=str(fold_log_dir))
        writer.close()

        # Collect metrics from results.csv if available
        metrics = {
            "fold": fold.fold_index,
            "mission_id": fold.mission_id,
            "weights_path": weights_path,
        }
        results_csv = Path(fold_log_dir) / "results.csv"
        if results_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(results_csv)
                if not df.empty:
                    metrics.update({
                        "final_map50": float(df.get("metrics/mAP50(B)").iloc[-1]) if "metrics/mAP50(B)" in df.columns else None,
                        "final_map5095": float(df.get("metrics/mAP50-95(B)").iloc[-1]) if "metrics/mAP50-95(B)" in df.columns else None,
                        "final_precision": float(df.get("metrics/precision(B)").iloc[-1]) if "metrics/precision(B)" in df.columns else None,
                        "final_recall": float(df.get("metrics/recall(B)").iloc[-1]) if "metrics/recall(B)" in df.columns else None,
                    })
            except Exception:
                pass

        # Placeholder for CV sweep results (to be populated by evaluation stage)
        metrics.setdefault("best_conf", None)
        metrics.setdefault("best_nms_iou", None)
        metrics.setdefault("best_metric", None)
        metrics.setdefault("epoch_best_saved", None)

        # Include fixed-conf per-epoch summary if available from latest epoch logging
        # We read back the most recent quick_eval JSON if present
        try:
            epoch_viz = sorted((Path(fold_log_dir) / "val_viz").glob("epoch_*/"))
            if epoch_viz:
                last_epoch_dir = epoch_viz[-1]
                # Not storing JSON per-epoch currently; we rely on TensorBoard scalars only
                # So we skip extracting values here to avoid coupling
                pass
        except Exception:
            pass

        # Perform operating point selection using config eval sweeps
        try:
            cfg_eval = cfg.eval
            # Load canonical val set for the fold
            samples = load_canonical_dataset(labels_json=cfg.data.labels_json, splits_dir=str(fold.fold_dir))
            val_samples = samples.val

            # Build predictions for each image for each NMS IoU setting
            nms_sweep = cfg_eval.nms_iou_sweep or []
            if not nms_sweep:
                nms_sweep = [None]

            best_overall = None
            best_overall_iou = None

            if cfg.model in ["yolo_v8", "yolo_nano"]:
                try:
                    from ultralytics import YOLO
                    yolo_model = YOLO(weights_path)
                except Exception:
                    yolo_model = None

                for iou_nms in nms_sweep:
                    per_image_predictions = []  # list of (scores, boxes_xyxy, H)
                    per_image_gt_boxes = []
                    for s in val_samples:
                        # Run prediction with minimal threshold to collect all
                        pred_kwargs = {"conf": 0.0, "verbose": False, "save": False}
                        if iou_nms is not None:
                            pred_kwargs["iou"] = float(iou_nms)
                        pred_kwargs["max_det"] = int(getattr(cfg_eval, "max_det", 1) or 1)
                        if yolo_model is not None:
                            r = yolo_model.predict(source=s.image_path, **pred_kwargs)
                            boxes = r[0].boxes.xyxy.cpu().numpy().tolist()
                            scores = r[0].boxes.conf.cpu().numpy().tolist()
                        else:
                            boxes, scores = [], []
                        per_image_predictions.append((scores, boxes, s.height, s.width))
                        per_image_gt_boxes.append(s.boxes)

                    sel = select_operating_point(
                        conf_sweep=cfg_eval.conf_sweep,
                        metric=cfg_eval.metric,
                        per_image_predictions=per_image_predictions,
                        per_image_gt_boxes=per_image_gt_boxes,
                        iou_thresh=cfg_eval.iou_thresh,
                        edge_k=cfg_eval.edge_touch_k,
                    )
                    score = sel.get("metrics", {}).get(cfg_eval.metric, 0.0) or 0.0
                    if (best_overall is None) or (score > (best_overall.get("metrics", {}).get(cfg_eval.metric, 0.0) or 0.0)):
                        best_overall = sel
                        best_overall_iou = iou_nms

            elif cfg.model == "faster_rcnn":
                # Build model and load weights
                try:
                    import torch
                    from src.trainers.rcnn_trainer import RCNNTrainer
                    r_trainer = RCNNTrainer(cfg)
                    r_trainer.model = r_trainer._initialize_model()
                    device = torch.device(cfg.device)
                    r_trainer.model.load_state_dict(torch.load(weights_path, map_location=device))
                    r_trainer.model.to(device)
                    r_trainer.model.eval()
                except Exception:
                    r_trainer = None

                for iou_nms in nms_sweep:
                    if r_trainer is not None and iou_nms is not None:
                        try:
                            # Adjust NMS threshold if available
                            if hasattr(r_trainer.model, "roi_heads") and hasattr(r_trainer.model.roi_heads, "nms_thresh"):
                                r_trainer.model.roi_heads.nms_thresh = float(iou_nms)
                        except Exception:
                            pass

                    per_image_predictions = []
                    per_image_gt_boxes = []
                    for s in val_samples:
                        if r_trainer is not None:
                            from PIL import Image
                            import torchvision.transforms as T
                            img = Image.open(s.image_path).convert("RGB")
                            transform = T.Compose([
                                T.Resize((cfg.rcnn.imgsz, cfg.rcnn.imgsz)),
                                T.ToTensor(),
                                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                            ])
                            img_tensor = transform(img).unsqueeze(0).to(device)
                            with torch.no_grad():
                                pred = r_trainer.model(img_tensor)[0]
                            boxes = pred['boxes'].cpu().numpy().tolist()
                            scores = pred['scores'].cpu().numpy().tolist()
                        else:
                            boxes, scores = [], []
                        per_image_predictions.append((scores, boxes, s.height, s.width))
                        per_image_gt_boxes.append(s.boxes)

                    sel = select_operating_point(
                        conf_sweep=cfg_eval.conf_sweep,
                        metric=cfg_eval.metric,
                        per_image_predictions=per_image_predictions,
                        per_image_gt_boxes=per_image_gt_boxes,
                        iou_thresh=cfg_eval.iou_thresh,
                        edge_k=cfg_eval.edge_touch_k,
                    )
                    score = sel.get("metrics", {}).get(cfg_eval.metric, 0.0) or 0.0
                    if (best_overall is None) or (score > (best_overall.get("metrics", {}).get(cfg_eval.metric, 0.0) or 0.0)):
                        best_overall = sel
                        best_overall_iou = iou_nms

            if best_overall is not None:
                metrics.update({
                    "best_conf": best_overall.get("best_conf"),
                    "best_nms_iou": best_overall_iou,
                    "best_metric": best_overall.get("metrics", {}).get(cfg_eval.metric),
                })
                # Log to TensorBoard (reopen writer to avoid closed handle)
                try:
                    tb = SummaryWriter(str(Path(fold_log_dir) / "tensorboard"))
                    per_t = best_overall.get("per_threshold", {})
                    for t, m in per_t.items():
                        tb.add_scalar(f"cv_sweep/{fold_name}/f1", m.get("f1", 0.0), int(round(float(t)*100)))
                        tb.add_scalar(f"cv_sweep/{fold_name}/precision", m.get("precision", 0.0), int(round(float(t)*100)))
                        tb.add_scalar(f"cv_sweep/{fold_name}/recall", m.get("recall", 0.0), int(round(float(t)*100)))
                    # Also log best threshold
                    if metrics.get("best_conf") is not None:
                        tb.add_scalar(f"val/best_threshold", float(metrics["best_conf"]), 0)
                    tb.close()
                except Exception:
                    pass
                # Save JSON
                # Include TP/FP/FN counts per threshold
                with open(Path(fold_log_dir) / "cv_operating_point.json", "w") as f:
                    json.dump({
                        "best": metrics,
                        "per_threshold": best_overall.get("per_threshold", {})
                    }, f, indent=2)
        except Exception:
            pass

        per_fold_metrics.append(metrics)

    # Aggregate
    def _agg(values: List[float]):
        vals = [v for v in values if isinstance(v, (int, float))]
        if not vals:
            return None, None
        if len(vals) == 1:
            return vals[0], 0.0
        return float(statistics.mean(vals)), float(statistics.pstdev(vals))

    agg = {}
    for key in ["final_map50", "final_map5095", "final_precision", "final_recall", "best_conf", "best_nms_iou"]:
        mean, std = _agg([m.get(key) for m in per_fold_metrics])
        agg[f"{key}_mean"] = mean
        agg[f"{key}_std"] = std

    # Write CSV summary
    try:
        import csv
        csv_path = Path(base_log_dir) / "summary" / "lomo_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "fold","mission_id","weights_path","final_map50","final_map5095","final_precision","final_recall","best_conf","best_nms_iou","best_metric","epoch_best_saved"
        ]
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            for row in per_fold_metrics:
                writer_csv.writerow({k: row.get(k) for k in fieldnames})
    except Exception:
        pass

    # Save summary JSON and CSV
    summary_dir = Path(base_log_dir) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "lomo_summary.json", "w") as f:
        json.dump({"per_fold": per_fold_metrics, "aggregate": agg}, f, indent=2)

    # Optional TensorBoard aggregate under summary/tensorboard
    try:
        writer = SummaryWriter(str(Path(summary_dir) / "tensorboard"))
        for k, v in agg.items():
            if v is not None:
                writer.add_scalar(k, v, 0)
        writer.close()
    except Exception:
        pass

    # Append compact & extended CSV for this CV run
    try:
        # Build run_info for CV
        # We cannot access cfg here directly; infer from last fold config stored on disk
        # Instead, capture minimal fields from first fold's persisted config.json
        run_id = base_name
        ts = cv_timestamp
        # Attempt to load a fold config to extract core fields
        model = ""
        imgsz = None
        batch_size = None
        epochs_planned = None
        lr0 = None
        lrf = None
        weight_decay = None
        momentum = None
        optimizer = None
        ema = None
        n_augmentations = None
        op_iou_thresh = None
        fp_rate_cap = None
        max_det = None
        edge_touch_k = None
        conf_desc = ""
        seed = None

        try:
            # Pick first fold directory
            fold_dirs = sorted(Path(base_log_dir).glob('fold_*'))
            if fold_dirs:
                with open(fold_dirs[0] / 'config.json', 'r') as cf:
                    cfg0 = json.load(cf)
                model = cfg0.get('model', '')
                seed = cfg0.get('seed')
                if 'yolo' in cfg0 and cfg0.get('model', '').startswith('yolo'):
                    y = cfg0['yolo']
                    imgsz = y.get('imgsz')
                    batch_size = y.get('batch_size')
                    epochs_planned = y.get('epochs')
                    lr0 = y.get('lr0')
                    lrf = y.get('lrf')
                    weight_decay = y.get('weight_decay')
                    momentum = y.get('momentum')
                    optimizer = y.get('optimizer')
                    ema = y.get('ema')
                    n_augmentations = y.get('n_augmentations')
                if 'rcnn' in cfg0 and cfg0.get('model') == 'faster_rcnn':
                    r = cfg0['rcnn']
                    imgsz = r.get('imgsz')
                    batch_size = r.get('batch_size')
                    epochs_planned = r.get('epochs')
                    lr0 = r.get('learning_rate')
                    weight_decay = r.get('weight_decay')
                    momentum = r.get('momentum')
                    optimizer = r.get('optimizer')
                if 'eval' in cfg0:
                    e = cfg0['eval']
                    op_iou_thresh = e.get('iou_thresh')
                    fp_rate_cap = e.get('fp_rate_cap')
                    max_det = e.get('max_det')
                    edge_touch_k = e.get('edge_touch_k')
                    cs = e.get('conf_sweep') or []
                    if cs:
                        try:
                            cmin = float(min(cs)); cmax = float(max(cs)); npts = len(cs)
                            conf_desc = f"{cmin:.3f}–{cmax:.3f}:{npts}pts"
                        except Exception:
                            conf_desc = ""
        except Exception:
            pass

        run_info = {
            'run_name': run_id,
            'run_id': run_id,
            'timestamp': ts,
            'git_commit': None,
            'model': model,
            'imgsz': imgsz,
            'batch_size': batch_size,
            'epochs_planned': epochs_planned,
            'lr0': lr0,
            'lrf': lrf,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'optimizer': optimizer,
            'scheduler': '',
            'ema': ema,
            'n_augmentations': n_augmentations,
            'op_iou_thresh': op_iou_thresh,
            'fp_rate_cap': fp_rate_cap,
            'max_det': max_det,
            'edge_touch_k': edge_touch_k,
            'conf_sweep_desc': conf_desc,
            'seed': seed,
        }

        # Folds metrics mapping to compact schema
        folds_metrics_comp: List[Dict[str, Any]] = []
        for m in per_fold_metrics:
            folds_metrics_comp.append({
                'f1_best': m.get('best_metric'),
                'mean_iou_tp_best': None,
                'map50_95': m.get('final_map5095'),
                'fp_rate_best': None,
                'epoch_best': m.get('epoch_best_saved'),
            })

        # Aggregate mapping
        aggregate_comp = {
            'f1_best_mean': agg.get('best_metric_mean') or None,
            'mean_iou_tp_best_mean': None,
            'map50_95_mean': agg.get('final_map5095_mean'),
            'map50_mean': agg.get('final_map50_mean'),
            'best_conf_mean': agg.get('best_conf_mean'),
            'fp_rate_best_mean': None,
            'best_epoch_mean': agg.get('epoch_best_saved_mean'),
        }

        # Paths
        paths = {
            'weights_best_path': None,
            'tb_dir': str(Path(summary_dir) / 'tensorboard'),
            'summary_json': str(summary_dir / 'lomo_summary.json'),
        }

        # time is not directly tracked here; leave empty in compact
        RunLogger.append_compact(run_info, folds_metrics_comp, aggregate_comp, None, fold_index=None)
        RunLogger.append_extended(run_info, aggregate_comp, paths)
    except Exception:
        pass

    return {"per_fold": per_fold_metrics, "aggregate": agg}


