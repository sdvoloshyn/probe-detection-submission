"""
YOLO trainer using Ultralytics with offline augmentation export.
"""
import os
import shutil
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from ultralytics import YOLO
ULTRALYTICS_AVAILABLE = True

from src.config import Config
from src.datasets.canonical import load_canonical_dataset
from src.datasets.adapters.yolo_export import export_yolo_dataset, cleanup_yolo_dataset
from src.utils.run_logger import RunLogger
# quick per-epoch eval removed; we use per-epoch full sweep policy


class TensorBoardCallback:
    """Custom callback for YOLO training to log metrics to TensorBoard."""
    
    def __init__(self, writer: SummaryWriter, log_interval: int = 1):
        self.writer = writer
        self.log_interval = log_interval
        self.epoch = 0
        
    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch."""
        if hasattr(trainer, 'metrics') and trainer.metrics:
            metrics = trainer.metrics
            
            # Log training losses
            if 'train/box_loss' in metrics:
                self.writer.add_scalar('Loss/Box_Loss', metrics['train/box_loss'], self.epoch)
            if 'train/cls_loss' in metrics:
                self.writer.add_scalar('Loss/Class_Loss', metrics['train/cls_loss'], self.epoch)
            if 'train/dfl_loss' in metrics:
                self.writer.add_scalar('Loss/DFL_Loss', metrics['train/dfl_loss'], self.epoch)
            
            # Core performance metrics from Ultralytics
            if 'metrics/mAP50(B)' in metrics:
                self.writer.add_scalar('Metrics/mAP50', metrics['metrics/mAP50(B)'], self.epoch)
            if 'metrics/mAP50-95(B)' in metrics:
                self.writer.add_scalar('Metrics/mAP50-95', metrics['metrics/mAP50-95(B)'], self.epoch)
            
            # Log learning rate
            if 'lr/pg0' in metrics:
                self.writer.add_scalar('Learning_Rate', metrics['lr/pg0'], self.epoch)
        
        self.epoch += 1
        self.writer.flush()


class YOLOTrainer:
    """YOLO trainer with offline augmentation export."""
    
    def __init__(self, config: Config, writer: Optional[SummaryWriter] = None, run_dir: Optional[str] = None):
        """
        Initialize YOLO trainer.
        
        Args:
            config: Configuration object
            writer: Optional TensorBoard writer for logging
            run_dir: Optional run directory to use instead of creating a new one
        """
        self.config = config
        self.model = None
        self.results = None
        self.writer = writer
        # Validate early stopping metric: only support 'f1_best' in the new policy
        es_metric_cfg = str(self.config.eval.early_stopping_metric).lower()
        if es_metric_cfg != 'f1_best':
            raise ValueError(f"Unsupported early_stopping_metric: {self.config.eval.early_stopping_metric}. Only 'f1_best' is supported.")
        self._epoch_idx = 0
        
        # Metric aliases no longer used
        self._metric_aliases = {}
        # Early-stopping state for per-epoch metrics
        self._es_best = -1.0
        self._es_no_improve = 0
        self._best_es_epoch: Optional[int] = None
        self._best_es_conf: Optional[float] = None
        self._best_es_metrics: Optional[Dict[str, float]] = None
        self._final_results = None
        self._val_paths: Optional[list[str]] = None
        self._labels_dict: Optional[dict] = None
        
        # Set up reproducibility
        self._set_seeds()
        
        # Disable Ultralytics Albumentations globally (no-op photometrics)
        self._disable_ultralytics_albumentations()
        
        # Create output directory
        if run_dir is not None:
            # Use provided run directory
            self.output_dir = Path(run_dir)
            self.run_name = self.output_dir.name
        else:
            # Create new run directory inside provided run_dir (from entrypoint)
            # Fallback to ./runs if none was provided
            base = Path.cwd() / 'runs'
            self.run_name = self._create_run_name()
            self.output_dir = base / self.run_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer if not provided
        if self.writer is None:
            tb_log_dir = self.output_dir / "tensorboard"
            self.writer = SummaryWriter(str(tb_log_dir))
            print(f"TensorBoard logs will be saved to: {tb_log_dir}")
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.config.seed)
        
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Set deterministic backends
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _resolve_device_string(self) -> str:
        """Resolve device string from config to a valid device string.
        Supports 'cpu', 'mps', 'cuda', 'cuda:0', or plain GPU index like '0'.
        """
        dev = str(self.config.device).strip().lower()
        if dev.isdigit():
            # Map '0' -> 'cuda:0' when CUDA is available
            if torch.cuda.is_available():
                return f"cuda:{dev}"
            return "cpu"
        if dev.startswith("cuda"):
            if torch.cuda.is_available():
                return dev
            return "cpu"
        if dev == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        # Default to CPU for 'cpu' or anything else
        return "cpu"
    
    def _disable_ultralytics_albumentations(self):
        """Disable Ultralytics Albumentations globally (no-op photometrics)."""
        try:
            from ultralytics.data import augment as _U
            if not hasattr(_U.Albumentations, '_probe_noop_installed'):
                def _no_albu(self, p: float = 1.0):
                    import albumentations as A
                    print(f"🔧 Albumentations monkey-patch called with p={p}")
                    self.p = p
                    self.contains_spatial = False
                    self.transform = A.Compose([])
                    print(f"   Set transform to empty Compose with {len(self.transform.transforms)} transforms")
                _U.Albumentations.__init__ = _no_albu
                _U.Albumentations._probe_noop_installed = True
                print("✅ Ultralytics Albumentations disabled (no-op)")
        except Exception as e:
            print(f"⚠️  Could not disable Ultralytics Albumentations: {e}")
    
    def _create_run_name(self) -> str:
        """Create run name with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"yolo_{self.config.model}_{timestamp}"
    
    def _infer_default_weights(self) -> str:
        name = str(self.config.model).lower().strip()
        # Special-case legacy names
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
    
    def train(self) -> str:
        """
        Train YOLO model.
        
        Returns:
            Path to trained model weights
        """
        _time_start = time.time()
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")
        
        print(f"🚀 YOLO TRAINER")
        print(f"Model: {self.config.model}")
        print(f"Run: {self.run_name}")
        
        # Load canonical dataset
        print("Loading canonical dataset...")
        samples = load_canonical_dataset(
            labels_json=self.config.data.labels_json,
            splits_dir=self.config.data.splits_dir
        )
        
        # Create unique temporary directory for this training run
        temp_dir = self.output_dir / "yolo_dataset"
        
        # Export to YOLO format with augmentation
        print("Exporting to YOLO format with augmentation...")
        # Set seed before augmentation to ensure deterministic results
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.dataset_yaml = export_yolo_dataset(
            samples=samples,
            config=self.config,
            output_dir=str(temp_dir),
            n_augmentations=self.config.yolo.n_augmentations,
            include_original=True
        )
        # Precompute val paths and labels_dict (pixel xyxy) once
        import yaml as _yaml
        ds_yaml_path = Path(self.dataset_yaml)
        with open(ds_yaml_path, 'r') as f:
            ds_cfg = _yaml.safe_load(f)
        root = Path(ds_cfg['path'])
        val_images_dir = root / ds_cfg['val']
        labels_dir = root / 'labels' / 'val'
        self._val_paths = [Path(p).resolve().as_posix() for p in sorted(Path(val_images_dir).glob('*.jpg'))]
        self._labels_dict = {}
        from PIL import Image as _Image
        for imgp in self._val_paths:
            lab = labels_dir / (Path(imgp).stem + '.txt')
            bboxes = []
            if lab.exists():
                _im = _Image.open(imgp)
                _W, _H = _im.size
                with open(lab, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            _, cx, cy, w, h = parts[:5]
                            cx = float(cx) * _W; cy = float(cy) * _H
                            w = float(w) * _W; h = float(h) * _H
                            x1 = max(0.0, cx - w/2.0); y1 = max(0.0, cy - h/2.0)
                            x2 = min(float(_W), cx + w/2.0); y2 = min(float(_H), cy + h/2.0)
                            bboxes.append({'xyxy': [x1, y1, x2, y2]})
            self._labels_dict[imgp] = {'bboxes': bboxes}
        
        # Initialize YOLO model with pretrained weights
        pretrained_weights = (self.config.yolo.weights or "").strip()
        if not pretrained_weights:
            pretrained_weights = self._infer_default_weights()
        
        print(f"Initializing YOLO model with pretrained weights: {pretrained_weights}")
        # Let Ultralytics handle the model initialization and weight downloading
        self.model = YOLO(pretrained_weights)
        
        # Ensure all parameters are trainable (including DFL head)
        for param in self.model.model.parameters():
            param.requires_grad = True
        print("All model parameters are trainable (including DFL head)")


        
        # Move model to device
        device_str = self._resolve_device_string()
        self.model.to(device_str)
        print(f"Moved model to {device_str}")
        
        # Set PyTorch's current device to match
        if device_str.startswith('cuda:'):
            device_id = int(device_str.split(':')[1])
            torch.cuda.set_device(device_id)
            print(f"Set PyTorch current device to {device_id}")
        
        # Prepare training arguments
        # Enable native YOLO geometric augmentations using values from config.aug
        # Keep photometric knobs at 0.0 (handled offline) and respect rect from config
        train_args = {
            'data': self.dataset_yaml,
            'epochs': self.config.yolo.epochs,
            'batch': self.config.yolo.batch_size,
            'imgsz': self.config.yolo.imgsz,
            'device': self.config.device,
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            'exist_ok': True,  # Do not auto-increment if directory exists (keep fold_X)
            'save': True,
            'save_period': 10,
            # We manage early stopping per-epoch using TB metrics; disable Ultralytics patience
            'patience': 0,
            'rect': self.config.yolo.rect,
            'workers': 4,  # Set appropriate number of workers for data loading
            'augment': False,  # Use native YOLO geometric augmentations
            'amp': False,  # Disable automatic mixed precision
            'mosaic': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'auto_augment': None,
            'erasing': 0.0,
            # Optimizer hyperparams
            'optimizer': self.config.yolo.optimizer,
            'lr0': self.config.yolo.lr0,
            'momentum': self.config.yolo.momentum,
            'weight_decay': self.config.yolo.weight_decay,
            'lrf': self.config.yolo.lrf,
            'cos_lr': self.config.yolo.cos_lr,
            'warmup_epochs': self.config.yolo.warmup_epochs,
            'warmup_momentum': self.config.yolo.warmup_momentum,
            'warmup_bias_lr': self.config.yolo.warmup_bias_lr,
            'degrees': float(self.config.aug.degrees),
            'translate': float(self.config.aug.translate),
            'scale': float(max(abs(self.config.aug.scale_max - 1.0), abs(1.0 - self.config.aug.scale_min))),
            'shear': float(self.config.aug.shear),
            'perspective': 0.0,
            'flipud': float(self.config.aug.flipud_p),
            'fliplr': float(self.config.aug.fliplr_p),
            'single_cls': True, 
            'verbose': True,
            'val': True,  # Enable validation for proper training loop
            'plots': True,  # Disable plotting to avoid KeyError with single class
            'deterministic': True,  # Enable deterministic training
            'seed': self.config.seed,  # Set YOLO seed
        }
        
        print("Starting YOLO training...")
        
        # Reset all seeds right before training for maximum determinism
        self._set_seeds()
        
        # Add TensorBoard callback and optional per-epoch eval if writer is available
        if self.writer is not None:
            try:
                tb_callback = TensorBoardCallback(self.writer)
                # Register TB callback
                if hasattr(self.model, 'add_callback'):
                    self.model.add_callback('on_train_epoch_end', tb_callback.on_train_epoch_end)
                print("TensorBoard logging enabled")
            except Exception:
                tb_callback = None
            
            def _on_epoch_end_eval(tr):
                # Increment epoch counter and run per-epoch full sweep
                self._epoch_idx += 1
                weights_last = Path(self.output_dir) / 'weights' / 'last.pt'
                if not weights_last.exists():
                    return
                from ultralytics import YOLO as _Y
                eval_model = _Y(str(weights_last))
                from src.eval.op_select import build_val_cache, select_operating_point
                pred_cache = build_val_cache(
                    model=eval_model,
                    val_paths=self._val_paths,
                    labels_dict=self._labels_dict,
                    imgsz=self.config.yolo.imgsz,
                    base_conf=0.001,
                    nms_iou=float(self.config.yolo.iou)
                )
                op = select_operating_point(
                    pred_cache=pred_cache,
                    labels_dict=self._labels_dict,
                    conf_sweep=self.config.eval.conf_sweep,
                    iou_thresh=float(self.config.eval.iou_thresh),
                    max_det=int(self.config.eval.max_det),
                    edge_touch_k=int(self.config.eval.edge_touch_k),
                    metric="f1",
                    fp_rate_cap=float(self.config.eval.fp_rate_cap)
                )
                if self.writer is not None:
                    self.writer.add_scalar('val_epoch/f1_best', float(op.best_metrics.f1), self._epoch_idx)
                    self.writer.add_scalar('val_epoch/best_conf', float(op.best_threshold), self._epoch_idx)
                    self.writer.add_scalar('val_epoch/fp_rate_best', float(op.best_metrics.fp_rate_neg), self._epoch_idx)
                    self.writer.add_scalar('val_epoch/recall_best', float(op.best_metrics.recall), self._epoch_idx)
                    self.writer.add_scalar('val_epoch/mean_iou_tp_best', float(op.best_metrics.mean_iou), self._epoch_idx)
                # Optionally save per-epoch sweep table
                if bool(self.config.eval.save_sweep_tables):
                    sweeps_dir = Path(self.output_dir) / 'val_sweeps'
                    sweeps_dir.mkdir(parents=True, exist_ok=True)
                    table = []
                    for m in op.all_thresholds:
                        table.append({
                            'conf': float(m.threshold),
                            'precision': float(m.precision),
                            'recall': float(m.recall),
                            'f1': float(m.f1),
                            'mean_iou_tp': float(m.mean_iou),
                            'fp_rate_neg': float(m.fp_rate_neg),
                            'tp': int(m.tp), 'fp': int(m.fp), 'fn': int(m.fn)
                        })
                    out_json = sweeps_dir / f"epoch_{self._epoch_idx:03d}.json"
                    import json as _json
                    with open(out_json, 'w') as f:
                        _json.dump(table, f, indent=2)
                    print(f"Saved per-epoch sweep to {out_json}")
                # Early stopping on F1@best
                if self.config.eval.es_min_epoch is not None and self._epoch_idx < int(self.config.eval.es_min_epoch):
                    return
                min_delta = float(self.config.eval.early_stopping_min_delta)
                if float(op.best_metrics.f1) > self._es_best + min_delta:
                    self._es_best = float(op.best_metrics.f1)
                    self._es_no_improve = 0
                    self._best_es_epoch = self._epoch_idx
                    self._best_es_conf = float(op.best_threshold)
                    self._best_es_metrics = {
                        'f1_best': float(op.best_metrics.f1),
                        'fp_rate_best': float(op.best_metrics.fp_rate_neg),
                        'recall_best': float(op.best_metrics.recall),
                        'mean_iou_tp_best': float(op.best_metrics.mean_iou)
                    }
                    # Snapshot current last.pt as best_es.pt for finalization
                    best_es_path = Path(self.output_dir) / 'weights' / 'best_es.pt'
                    shutil.copy2(weights_last, best_es_path)
                else:
                    self._es_no_improve += 1
                if self._es_no_improve >= int(self.config.eval.early_stopping_patience):
                    tr.stop_training = True
                    tr.stop = True
                    self.model.trainer.stop_training = True
                    self.model.trainer.stop = True
                    print(f"⏹️ Early stopping (F1@best) at epoch {self._epoch_idx}")
            try:
                if hasattr(self.model, 'add_callback'):
                    self.model.add_callback('on_train_epoch_end', _on_epoch_end_eval)
            except Exception:
                pass
        
        self.results = self.model.train(**train_args)
        
        # Post-process training results for TensorBoard
        if self.writer is not None:
            self._log_training_results()
        
        # Determine best weights path: prefer the snapshot we took at the best early-stopping epoch
        save_dir = Path(self.results.save_dir)
        best_es = save_dir / "weights" / "best_es.pt"
        best_weights = best_es if best_es.exists() else (save_dir / "weights" / "best.pt")

        weights_path = str(best_weights)
        
        # Run evaluation and artifact generation using best weights
        self._evaluate_with_weights(samples, weights_path)
        
        # Final visualization at locked OP
        if bool(self.config.eval.final_viz) and self._best_es_conf is not None:
            from src.eval.op_select import build_val_cache
            from src.eval.viz import save_eval_viz_from_cache
            eval_model = YOLO(weights_path)
            pred_cache_all = build_val_cache(
                model=eval_model,
                val_paths=self._val_paths,
                labels_dict=self._labels_dict,
                imgsz=self.config.yolo.imgsz,
                base_conf=0.001,
                nms_iou=self.config.yolo.iou
            )
            # Filter to final_best_conf and max_det=1 with edge rule
            filtered = {}
            from src.eval.op_select import Detection
            for imgp, dets in pred_cache_all.items():
                ds = [d for d in dets if d.conf >= float(self._best_es_conf)]
                if int(self.config.eval.edge_touch_k) > 0:
                    from PIL import Image as _Image
                    W, H = _Image.open(imgp).size
                    ds = [dd for dd in ds if dd.touches_edge(W, H, int(self.config.eval.edge_touch_k))]
                ds = sorted(ds, key=lambda d: d.conf, reverse=True)[:int(self.config.eval.max_det)]
                filtered[imgp] = ds
            viz_final = Path(self.output_dir) / 'val_viz_final'
            save_eval_viz_from_cache(filtered, self._labels_dict, str(viz_final), class_name='probe')
            print(f"Saved final overlays to {viz_final}/")
        
        # Write compact best.meta.json and final viz at chosen best epoch/conf
        if self._best_es_metrics is not None and self._best_es_conf is not None and self._best_es_epoch is not None:
            meta = {
                "conf": float(self._best_es_conf),
                "iou": float(self.config.eval.iou_thresh),
                "max_det": int(self.config.eval.max_det),
                "edge_touch_k": int(self.config.eval.edge_touch_k),
                **self._best_es_metrics,
                "epoch": int(self._best_es_epoch)
            }
            with open(Path(self.output_dir) / 'weights' / 'best.meta.json', 'w') as f:
                import json as _json
                _json.dump(meta, f, indent=2)
            # Also write summary.json in val_viz_final/
            try:
                viz_final = Path(self.output_dir) / 'val_viz_final'
                viz_final.mkdir(parents=True, exist_ok=True)
                with open(viz_final / 'summary.json', 'w') as f:
                    _json.dump(meta, f, indent=2)
                print(f"Wrote best.meta.json and {viz_final}/summary.json")
            except Exception:
                pass
        
        # Clean up temporary files
        cleanup_yolo_dataset(str(temp_dir))
        
        # Close TensorBoard writer
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
            print(f"TensorBoard logs saved to: {self.output_dir / 'tensorboard'}")
        
        # After successful training/evaluation, append CSV logs
        try:
            # Build folds_metrics (single-split → fold1 only)
            folds_metrics = []
            if self._best_es_metrics is not None:
                fm: Dict[str, Any] = {
                    'f1_best': float(self._best_es_metrics.get('f1_best', 0.0)),
                    'mean_iou_tp_best': float(self._best_es_metrics.get('mean_iou_tp_best', 0.0)),
                    'map50_95': float(self._final_results.box.map) if self._final_results is not None else None,
                    'fp_rate_best': float(self._best_es_metrics.get('fp_rate_best', 0.0)),
                    'best_conf': float(self._best_es_conf) if self._best_es_conf is not None else None,
                }
                folds_metrics.append(fm)

            # Aggregate for extended CSV
            aggregate: Dict[str, Any] = {}
            if folds_metrics:
                fm0 = folds_metrics[0]
                aggregate.update({
                    'f1_best_mean': fm0.get('f1_best'),
                    'mean_iou_tp_best_mean': fm0.get('mean_iou_tp_best'),
                    'map50_95_mean': fm0.get('map50_95'),
                    'map50_mean': float(self._final_results.box.map50) if self._final_results is not None else None,
                    'best_conf_mean': float(self._best_es_conf) if self._best_es_conf is not None else None,
                    'fp_rate_best_mean': fm0.get('fp_rate_best'),
                    'best_epoch_mean': int(self._best_es_epoch) if self._best_es_epoch is not None else None,
                })

            # Run identity and config
            eval_cfg = self.config.eval
            yolo_cfg = self.config.yolo
            confs = list(eval_cfg.conf_sweep or [])
            conf_desc = ""
            if confs:
                try:
                    cmin = float(min(confs)); cmax = float(max(confs)); npts = len(confs)
                    conf_desc = f"{cmin:.3f}–{cmax:.3f}:{npts}pts"
                except Exception:
                    conf_desc = ""
            # If cv_context.json exists (from LOMO runner), reuse its run_name+timestamp
            try:
                import json as _json
                cv_ctx_path = self.output_dir / 'cv_context.json'
                cv_ctx = None
                if cv_ctx_path.exists():
                    with open(cv_ctx_path, 'r') as f:
                        cv_ctx = _json.load(f)
            except Exception:
                cv_ctx = None
            run_info = {
                'run_name': (cv_ctx.get('run_name') if cv_ctx else self.run_name),
                'run_id': (cv_ctx.get('run_name') if cv_ctx else self.run_name),
                'timestamp': (cv_ctx.get('timestamp') if cv_ctx else datetime.now().astimezone().isoformat()),
                'git_commit': None,
                'model': self.config.model,
                'imgsz': yolo_cfg.imgsz,
                'batch_size': yolo_cfg.batch_size,
                'epochs_planned': yolo_cfg.epochs,
                'lr0': yolo_cfg.lr0,
                'lrf': yolo_cfg.lrf,
                'weight_decay': yolo_cfg.weight_decay,
                'momentum': yolo_cfg.momentum,
                'optimizer': yolo_cfg.optimizer,
                'scheduler': 'cosine' if bool(yolo_cfg.cos_lr) else '',
                'ema': yolo_cfg.ema,
                'n_augmentations': yolo_cfg.n_augmentations,
                'op_iou_thresh': eval_cfg.iou_thresh,
                'fp_rate_cap': eval_cfg.fp_rate_cap,
                'max_det': eval_cfg.max_det,
                'edge_touch_k': eval_cfg.edge_touch_k,
                'conf_sweep_desc': conf_desc,
                'seed': self.config.seed,
                'n_augmentations': yolo_cfg.n_augmentations,
            }

            # Paths for extended
            save_dir = Path(self.results.save_dir)
            best_meta = save_dir / 'weights' / 'best.meta.json'
            summary_json = self.output_dir / 'op_select_summary.json'
            paths = {
                'weights_best_path': weights_path,
                'tb_dir': str(self.output_dir / 'tensorboard'),
                'summary_json': str(summary_json if summary_json.exists() else best_meta),
            }

            time_elapsed = time.time() - _time_start
            # include epoch_best in folds_metrics/aggregate if available
            if self._best_es_epoch is not None:
                if folds_metrics:
                    folds_metrics[0]['epoch_best'] = float(self._best_es_epoch)
                aggregate['epoch_best_mean'] = float(self._best_es_epoch)
            # determine fold index from cv_context if present; else default to 1
            fold_idx = 1
            if cv_ctx and isinstance(cv_ctx.get('fold_index'), int):
                fold_idx = int(cv_ctx['fold_index'])
            RunLogger.append_compact(run_info, folds_metrics, aggregate, time_elapsed, fold_index=fold_idx)
            RunLogger.append_extended(run_info, aggregate, paths)
        except Exception as _e:
            print(f"⚠️  Run logging failed: {_e}")

        print(f"✅ Training complete! Weights saved to: {weights_path}")
        return weights_path
    
    def _log_training_results(self):
        """Log training results from CSV file to TensorBoard."""
        if self.writer is None:
            return
            
        try:
            import pandas as pd
            
            # Look for results.csv in the training output directory
            results_file = Path(self.results.save_dir) / "results.csv"
            if results_file.exists():
                df = pd.read_csv(results_file)
                
                for epoch, row in df.iterrows():
                    # Log training losses
                    if 'train/box_loss' in row:
                        self.writer.add_scalar('Loss/Box_Loss', row['train/box_loss'], epoch + 1)
                    if 'train/cls_loss' in row:
                        self.writer.add_scalar('Loss/Class_Loss', row['train/cls_loss'], epoch + 1)
                    if 'train/dfl_loss' in row:
                        self.writer.add_scalar('Loss/DFL_Loss', row['train/dfl_loss'], epoch + 1)
                    
                    # Core performance metrics
                    if 'metrics/mAP50(B)' in row:
                        self.writer.add_scalar('Metrics/mAP50', row['metrics/mAP50(B)'], epoch + 1)
                    if 'metrics/mAP50-95(B)' in row:
                        self.writer.add_scalar('Metrics/mAP50-95', row['metrics/mAP50-95(B)'], epoch + 1)
                    
                    # Log learning rate
                    if 'lr/pg0' in row:
                        self.writer.add_scalar('Learning_Rate', row['lr/pg0'], epoch + 1)
                
                print(f"✅ Logged {len(df)} epochs to TensorBoard")
            else:
                print("⚠️  No results.csv found for TensorBoard logging")
                
        except Exception as e:
            print(f"⚠️  Error logging to TensorBoard: {e}")
    
    def _evaluate_with_weights(self, samples, weights_path: str):
        """Run unified operating point selection on validation set using specified weights."""
        from src.eval.op_select import build_val_cache, select_operating_point, save_operating_point_results, save_best_metadata
        
        print("Running unified operating point selection...")
        
        # Get evaluation config - require it to be present
        if not hasattr(self.config, 'eval') or self.config.eval is None:
            raise ValueError("❌ eval configuration is required but not found in config.yaml")
        
        eval_config = self.config.eval
        conf_sweep = eval_config.conf_sweep
        iou_thresh = eval_config.iou_thresh
        max_det = eval_config.max_det
        edge_touch_k = eval_config.edge_touch_k
        fp_rate_cap = eval_config.fp_rate_cap
        
        # Load validation data (prefer precomputed canonicalized paths/labels)
        if self._val_paths is not None and self._labels_dict is not None:
            val_paths = self._val_paths
            labels_dict = self._labels_dict
        else:
            val_paths = [sample.image_path for sample in samples.val]
            labels_dict = {}
            for sample in samples.val:
                labels_dict[sample.image_path] = {
                    'bboxes': [{'xyxy': box} for box in sample.boxes]
                }
        
        # Load best weights into a fresh model for evaluation
        eval_model = YOLO(weights_path)
        # Build validation cache
        pred_cache = build_val_cache(
            model=eval_model,
            val_paths=val_paths,
            labels_dict=labels_dict,
            imgsz=self.config.yolo.imgsz,
            base_conf=0.001,
            nms_iou=self.config.yolo.iou
        )
        
        # Select operating point
        op_point = select_operating_point(
            pred_cache=pred_cache,
            labels_dict=labels_dict,
            conf_sweep=conf_sweep,
            iou_thresh=iou_thresh,
            max_det=max_det,
            edge_touch_k=edge_touch_k,
            metric="f1",
            fp_rate_cap=fp_rate_cap
        )
        print(f"✅ Selected confidence threshold: {float(op_point.best_threshold):.3f}")
        
        # Save results
        save_operating_point_results(op_point, self.output_dir)
        
        # Generate validation visualizations
        from src.eval.viz import save_eval_viz_from_cache
        viz_dir = Path(self.output_dir) / "val_viz"
        save_eval_viz_from_cache(pred_cache, labels_dict, viz_dir, class_name="probe")
        
        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('val/best_conf', float(op_point.best_threshold), 0)
            self.writer.add_scalar('val/f1@best', op_point.best_metrics.f1, 0)
            self.writer.add_scalar('val/precision@best', op_point.best_metrics.precision, 0)
            self.writer.add_scalar('val/recall@best', op_point.best_metrics.recall, 0)
            self.writer.add_scalar('val/mean_iou@best', op_point.best_metrics.mean_iou, 0)
        
        # Run final validation with selected threshold
        print(f"\nFinal YOLO Validation Results (conf={float(op_point.best_threshold):.3f}):")
        final_results = eval_model.val(
            data=self.dataset_yaml,
            split='val',
            conf=float(op_point.best_threshold),
            save_json=True,
            save_hybrid=True
        )
        
        # Store final results for CSV logging
        self._final_results = final_results
        
        # Save metadata alongside weights
        save_dir = Path(self.results.save_dir)
        weights_path = save_dir / "weights" / "best.pt"
        if weights_path.exists():
            additional_config = {
                "nms_iou": self.config.yolo.iou,
                "max_det": max_det,
                "edge_touch_k": edge_touch_k,
                "model_type": "yolo"
            }
            save_best_metadata(op_point, weights_path, additional_config)
        
        # Print results
        print(f"  Precision: {float(final_results.box.p[0]):.4f}")
        print(f"  Recall: {float(final_results.box.r[0]):.4f}")
        print(f"  F1: {float(final_results.box.f1[0]):.4f}")
        print(f"  AP@0.5: {float(final_results.box.map50):.4f}")
        print(f"  AP@0.75: {float(final_results.box.map75):.4f}")
        print(f"  AP@0.5:0.95: {float(final_results.box.map):.4f}")
        
        # Log to TensorBoard if available
        if self.writer is not None:
            self.writer.add_scalar('val/best_threshold', op_point.best_threshold, 0)
            self.writer.add_scalar('val/f1@best', float(final_results.box.f1[0]), 0)
            self.writer.add_scalar('val/precision@best', float(final_results.box.p[0]), 0)
            self.writer.add_scalar('val/recall@best', float(final_results.box.r[0]), 0)
            self.writer.add_scalar('val/map50@best', float(final_results.box.map50), 0)
            self.writer.add_scalar('val/map50_95@best', float(final_results.box.map), 0)
    
    def predict(self, image_path: str, conf: float = None) -> Dict[str, Any]:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to image
            conf: Confidence threshold
            
        Returns:
            Prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        conf = conf or self.config.yolo.conf or 0.5  # Fallback to 0.5 if no threshold set
        
        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=False,
            verbose=False
        )
        
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        scores = results[0].boxes.conf.cpu().numpy().tolist()
        labels = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
        # Enforce max_det from eval config
        try:
            max_det = int(getattr(self.config, 'eval').max_det)
        except Exception:
            max_det = 1
        if len(scores) > max_det:
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_det]
            boxes = [boxes[i] for i in order]
            scores = [scores[i] for i in order]
            labels = [labels[i] for i in order]
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {}
        
        return {
            'model_name': self.config.model,
            'weights_path': str(self.output_dir / "weights" / "best.pt"),
            'input_size': self.config.yolo.imgsz,
            'device': self.config.device,
            'run_name': self.run_name,
            'output_dir': str(self.output_dir)
        }
