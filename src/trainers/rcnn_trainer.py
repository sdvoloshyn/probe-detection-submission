"""
Faster R-CNN trainer with standard PyTorch training loop.
"""
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
from datetime import datetime
import json

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
TORCHVISION_AVAILABLE = True
from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.eval.filters import edge_touch_filter
from src.datasets.canonical import load_canonical_dataset
from src.datasets.augment import build_augmentation_pipelines
from src.datasets.adapters.rcnn_dataset import create_rcnn_data_loaders
from src.eval.validate import quick_evalFixedConf, run_validation
from src.utils.run_logger import RunLogger


class RCNNTrainer:
    """Faster R-CNN trainer with standard PyTorch training loop."""
    
    def __init__(self, config: Config, run_dir: Optional[str] = None):
        """
        Initialize RCNN trainer.
        
        Args:
            config: Configuration object
            run_dir: Optional run directory to use instead of creating a new one
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.best_metrics = None
        self.training_history = []
        self.writer = None  # Optional TensorBoard writer injected by caller
        # Early-stopping state (match YOLO policy: F1@best)
        self._epoch_idx = 0
        self._es_best: float = -1.0
        self._es_no_improve: int = 0
        self._best_es_epoch: Optional[int] = None
        self._best_es_conf: Optional[float] = None
        self._best_es_metrics: Optional[Dict[str, float]] = None
        
        # Set up reproducibility
        self._set_seeds()
        
        # Create output directory
        if run_dir is not None:
            # Use provided run directory
            self.output_dir = Path(run_dir)
            self.run_name = self.output_dir.name
        else:
            # Create new run directory inside ./runs by default
            base = Path.cwd() / 'runs'
            self.run_name = self._create_run_name()
            self.output_dir = base / self.run_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
    
    def _create_run_name(self) -> str:
        """Create run name with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"rcnn_{self.config.model}_{timestamp}"
    
    def _initialize_model(self) -> torch.nn.Module:
        """Initialize Faster R-CNN model."""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision not available. Install with: pip install torchvision")
        
        # Load pre-trained model based on backbone
        if self.config.rcnn.backbone == "resnet50_fpn":
            model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        elif self.config.rcnn.backbone == "mobilenet_v3_large_fpn":
            model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
        else:
            # Default to ResNet50 FPN if unknown backbone
            model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        
        # Replace classifier for single class (probe)
        num_classes = len(self.config.data.class_names) + 1  # +1 for background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # Apply RCNN-configurable model params
        try:
            from torchvision.models.detection.rpn import AnchorGenerator
            # Use existing generator to infer feature levels
            default_sizes = getattr(model.rpn.anchor_generator, 'sizes', tuple((32,), (64,), (128,), (256,), (512,)))
            n_levels = len(default_sizes) if isinstance(default_sizes, (list, tuple)) else 5
            # Build sizes as Tuple[Tuple[int]] with len == n_levels
            if self.config.rcnn.anchor_sizes:
                size_tuple = tuple(int(s) for s in self.config.rcnn.anchor_sizes)
                # If user specified one size per level
                if len(size_tuple) == n_levels:
                    sizes_new = tuple((s,) for s in size_tuple)
                else:
                    # Replicate same size set across all levels
                    sizes_new = tuple((size_tuple))
                    sizes_new = tuple(sizes_new for _ in range(n_levels))
            else:
                sizes_new = tuple(tuple(s) for s in default_sizes)

            # Build aspect ratios as Tuple[Tuple[float]] with len == n_levels
            ars_default = getattr(model.rpn.anchor_generator, 'aspect_ratios', ((0.5, 1.0, 2.0),) * n_levels)
            if self.config.rcnn.anchor_aspect_ratios:
                ar_tuple = tuple(float(r) for r in self.config.rcnn.anchor_aspect_ratios)
                ars_new = tuple(ar_tuple for _ in range(n_levels))
            else:
                # Preserve defaults
                ars_new = tuple(tuple(ar) for ar in ars_default)

            ag = AnchorGenerator(sizes=sizes_new, aspect_ratios=ars_new)
            model.rpn.anchor_generator = ag
        except Exception:
            pass
        # RPN thresholds/topK
        try:
            if self.config.rcnn.rpn_nms_thresh is not None:
                model.rpn.nms_thresh = float(self.config.rcnn.rpn_nms_thresh)
            if self.config.rcnn.rpn_pre_nms_topk_train is not None:
                model.rpn.pre_nms_top_n['training'] = int(self.config.rcnn.rpn_pre_nms_topk_train)
            if self.config.rcnn.rpn_pre_nms_topk_test is not None:
                model.rpn.pre_nms_top_n['testing'] = int(self.config.rcnn.rpn_pre_nms_topk_test)
            if self.config.rcnn.rpn_post_nms_topk_train is not None:
                model.rpn.post_nms_top_n['training'] = int(self.config.rcnn.rpn_post_nms_topk_train)
            if self.config.rcnn.rpn_post_nms_topk_test is not None:
                model.rpn.post_nms_top_n['testing'] = int(self.config.rcnn.rpn_post_nms_topk_test)
            if self.config.rcnn.rpn_fg_iou_thresh is not None:
                model.rpn.box_fg_iou_thresh = float(self.config.rcnn.rpn_fg_iou_thresh)
            if self.config.rcnn.rpn_bg_iou_thresh is not None:
                model.rpn.box_bg_iou_thresh = float(self.config.rcnn.rpn_bg_iou_thresh)
        except Exception:
            pass
        # ROI heads / box settings
        try:
            if self.config.rcnn.box_score_thresh is not None and hasattr(model.roi_heads, 'score_thresh'):
                model.roi_heads.score_thresh = float(self.config.rcnn.box_score_thresh)
            if self.config.rcnn.box_nms_thresh is not None and hasattr(model.roi_heads, 'nms_thresh'):
                model.roi_heads.nms_thresh = float(self.config.rcnn.box_nms_thresh)
            if self.config.rcnn.box_fg_iou_thresh is not None and hasattr(model.roi_heads, 'box_fg_iou_thresh'):
                model.roi_heads.box_fg_iou_thresh = float(self.config.rcnn.box_fg_iou_thresh)
            if self.config.rcnn.box_bg_iou_thresh is not None and hasattr(model.roi_heads, 'box_bg_iou_thresh'):
                model.roi_heads.box_bg_iou_thresh = float(self.config.rcnn.box_bg_iou_thresh)
            if self.config.rcnn.detections_per_img is not None and hasattr(model.roi_heads, 'detections_per_img'):
                model.roi_heads.detections_per_img = int(self.config.rcnn.detections_per_img)
        except Exception:
            pass
        # Enforce deployment policy: external thresholding & NMS (fallbacks)
        try:
            max_det = int(getattr(self.config, 'eval').max_det)
        except Exception:
            max_det = 1
        if hasattr(model.roi_heads, 'detections_per_img'):
            model.roi_heads.detections_per_img = max_det
        # Let our external sweep control conf threshold → make internal threshold permissive
        try:
            if hasattr(model.roi_heads, 'score_thresh'):
                model.roi_heads.score_thresh = 0.0
        except Exception:
            pass
        # Align NMS IoU with config.rcnn.iou if available
        try:
            nms_iou = float(getattr(self.config.rcnn, 'iou', 0.5))
            if hasattr(model.roi_heads, 'nms_thresh'):
                model.roi_heads.nms_thresh = nms_iou
        except Exception:
            pass
        
        return model

    def _resolve_device(self) -> torch.device:
        """Resolve device string from config to a valid torch.device.
        Supports 'cpu', 'mps', 'cuda', 'cuda:0', or plain GPU index like '0'.
        """
        dev = str(self.config.device).strip().lower()
        if dev.isdigit():
            # Map '0' -> 'cuda:0' when CUDA is available
            if torch.cuda.is_available():
                return torch.device(f"cuda:{dev}")
            return torch.device("cpu")
        if dev.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(dev)
            return torch.device("cpu")
        if dev == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        # Default to CPU for 'cpu' or anything else
        return torch.device("cpu")
    
    def train(self) -> str:
        """
        Train Faster R-CNN model.
        
        Returns:
            Path to trained model weights
        """
        _time_start = time.time()
        print(f"🚀 RCNN TRAINER")
        print(f"Model: {self.config.model}")
        print(f"Run: {self.run_name}")
        
        # Load canonical dataset
        print("Loading canonical dataset...")
        samples = load_canonical_dataset(
            labels_json=self.config.data.labels_json,
            splits_dir=self.config.data.splits_dir
        )
        
        # Build augmentation pipelines
        print("Building augmentation pipelines...")
        train_pipeline, val_pipeline = build_augmentation_pipelines(self.config.aug, imgsz=self.config.rcnn.imgsz)
        
        # Create data loaders
        print("Creating data loaders...")
        data_loaders = create_rcnn_data_loaders(
            samples={'train': samples.train, 'val': samples.val},
            augmentation_pipelines={'train': train_pipeline, 'val': val_pipeline},
            config=self.config.rcnn,
            num_workers=0  # Use 0 workers for MPS stability
        )
        
        # Initialize model
        print("Initializing Faster R-CNN model...")
        self.model = self._initialize_model()
        
        # Move model to device
        device = self._resolve_device()
        self.model.to(device)
        print(f"Moved model to {device}")
        
        # Initialize optimizer and scheduler from config
        opt_name = (self.config.rcnn.optimizer or 'SGD').lower()
        if opt_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.rcnn.learning_rate,
                momentum=self.config.rcnn.momentum,
                weight_decay=self.config.rcnn.weight_decay
            )
        elif opt_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.rcnn.learning_rate,
                weight_decay=self.config.rcnn.weight_decay
            )
        elif opt_name == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config.rcnn.learning_rate,
                momentum=self.config.rcnn.momentum,
                weight_decay=self.config.rcnn.weight_decay
            )
        else:  # default AdamW
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.rcnn.learning_rate,
                weight_decay=self.config.rcnn.weight_decay
            )
        
        # Gentle scheduler; large step with small gamma can explode LR early
        self.scheduler = StepLR(self.optimizer, step_size=max(10, int(self.config.rcnn.epochs)), gamma=0.5)
        
        # Prepare TensorBoard writer (if not injected by caller)
        if self.writer is None:
            tb_dir = self.output_dir / "tensorboard"
            self.writer = SummaryWriter(str(tb_dir))
            print(f"TensorBoard logs will be saved to: {tb_dir}")

        # Training loop
        print("Starting training...")
        patience_counter = 0
        
        for epoch in range(self.config.rcnn.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.rcnn.epochs}")
            
            # Training phase
            train_metrics = self._train_epoch(data_loaders['train'], device)
            
            # Validation phase with confidence sweep (TensorBoard + JSON)
            val_summary = run_validation(
                model=self.model,
                dataloader=data_loaders['val'],
                epoch=epoch + 1,
                writer=self.writer,
                config=self.config,
                save_dir=str(self.output_dir)
            )
            # Update current best operating point in config for inference convenience
            if val_summary and 'best_threshold' in val_summary:
                try:
                    self.config.rcnn.conf = float(val_summary['best_threshold'])
                except Exception:
                    pass
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'val_precision': float(val_summary['best_metrics']['precision']) if val_summary else 0.0,
                'val_recall': float(val_summary['best_metrics']['recall']) if val_summary else 0.0,
                'val_f1': float(val_summary['best_metrics']['f1']) if val_summary else 0.0,
                'val_mean_iou_tp': float(val_summary['best_metrics']['mean_iou']) if val_summary else 0.0,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_metrics)
            
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Precision: {epoch_metrics['val_precision']:.4f}")
            print(f"Val Recall: {epoch_metrics['val_recall']:.4f}")
            print(f"Val F1: {epoch_metrics['val_f1']:.4f}")
            
            # TensorBoard scalars
            try:
                self.writer.add_scalar('train/loss', float(train_metrics['loss']), epoch + 1)
                self.writer.add_scalar('train/lr', float(self.optimizer.param_groups[0]['lr']), epoch + 1)
            except Exception:
                pass

            # Early stopping on F1@best (match YOLO policy)
            self._epoch_idx += 1
            es_min_epoch = getattr(self.config.eval, 'es_min_epoch', None)
            can_consider = True
            if es_min_epoch is not None:
                try:
                    can_consider = (self._epoch_idx >= int(es_min_epoch))
                except Exception:
                    can_consider = True
            if val_summary and can_consider:
                cur_f1 = float(val_summary['best_metrics']['f1'])
                cur_conf = float(val_summary['best_threshold'])
                min_delta = float(getattr(self.config.eval, 'early_stopping_min_delta', 0.001))
                if cur_f1 > self._es_best + min_delta:
                    self._es_best = cur_f1
                    self._es_no_improve = 0
                    patience_counter = 0
                    self._best_es_epoch = self._epoch_idx
                    self._best_es_conf = cur_conf
                    self._best_es_metrics = {
                        'f1_best': float(val_summary['best_metrics']['f1']),
                        'precision_best': float(val_summary['best_metrics']['precision']),
                        'recall_best': float(val_summary['best_metrics']['recall']),
                        'mean_iou_tp_best': float(val_summary['best_metrics']['mean_iou'])
                    }
                    # Save snapshot of current best weights and metadata
                    self._save_model("best_model.pth")
                    try:
                        best_meta = {
                            'conf': float(self._best_es_conf),
                            'iou': float(getattr(self.config.eval, 'iou_thresh', 0.5)),
                            'max_det': int(getattr(self.config.eval, 'max_det', 1)),
                            'edge_touch_k': int(getattr(self.config.eval, 'edge_touch_k', 0)),
                            'epoch': int(self._best_es_epoch),
                            **{
                                'f1': float(val_summary['best_metrics']['f1']),
                                'precision': float(val_summary['best_metrics']['precision']),
                                'recall': float(val_summary['best_metrics']['recall']),
                                'mean_iou': float(val_summary['best_metrics']['mean_iou'])
                            }
                        }
                        with open(self.output_dir / 'best.meta.json', 'w') as f:
                            json.dump({'metrics': best_meta, 'conf': best_meta['conf']}, f, indent=2)
                    except Exception:
                        pass
                    print(f"✅ New best model saved (F1@best: {self._es_best:.4f}, conf={self._best_es_conf:.3f})")
                else:
                    self._es_no_improve += 1
                    patience_counter += 1
            
            # Early stopping
            if patience_counter >= int(getattr(self.config.rcnn, 'patience', 20)):
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        
        # Save final model
        self._save_model("final_model.pth")
        
        # Save training history
        self._save_training_history()
        
        # Run final evaluation
        self._final_evaluation(data_loaders['val'], device)
        
        weights_path = self.output_dir / "best_model.pth"
        # Append logs (single-split semantics)
        try:
            # If cv_context.json exists in run_dir, reuse its run_name+timestamp to upsert into same compact row
            try:
                import json as _json
                cv_ctx_path = self.output_dir / 'cv_context.json'
                cv_ctx = None
                if cv_ctx_path.exists():
                    with open(cv_ctx_path, 'r') as f:
                        cv_ctx = _json.load(f)
            except Exception:
                cv_ctx = None
            # Attempt to read best.meta.json for unified metrics if present
            best_meta_path = self.output_dir / 'best.meta.json'
            folds_metrics: List[Dict[str, Any]] = []
            aggregate: Dict[str, Any] = {}
            if best_meta_path.exists():
                with open(best_meta_path, 'r') as f:
                    meta = json.load(f)
                met = meta.get('metrics', {})
                folds_metrics.append({
                    'f1_best': float(met.get('f1', 0.0) or 0.0),
                    'mean_iou_tp_best': float(met.get('mean_iou', 0.0) or 0.0),
                    'map50_95': None,
                    'fp_rate_best': float(met.get('fp_rate_neg', 0.0) or 0.0),
                })
                aggregate.update({
                    'f1_best_mean': folds_metrics[0]['f1_best'],
                    'mean_iou_tp_best_mean': folds_metrics[0]['mean_iou_tp_best'],
                    'best_conf_mean': float(meta.get('conf')) if meta.get('conf') is not None else None,
                    'fp_rate_best_mean': folds_metrics[0]['fp_rate_best'],
                })
            # Build run_info
            eval_cfg = self.config.eval
            rcnn_cfg = self.config.rcnn
            confs = list(eval_cfg.conf_sweep or [])
            conf_desc = ""
            if confs:
                try:
                    cmin = float(min(confs)); cmax = float(max(confs)); npts = len(confs)
                    conf_desc = f"{cmin:.3f}–{cmax:.3f}:{npts}pts"
                except Exception:
                    conf_desc = ""
            run_info = {
                'run_name': (cv_ctx.get('run_name') if cv_ctx else self.run_name),
                'run_id': (cv_ctx.get('run_name') if cv_ctx else self.run_name),
                'timestamp': (cv_ctx.get('timestamp') if cv_ctx else datetime.now().astimezone().isoformat()),
                'git_commit': None,
                'model': self.config.model,
                'imgsz': rcnn_cfg.imgsz,
                'batch_size': rcnn_cfg.batch_size,
                'epochs_planned': rcnn_cfg.epochs,
                'lr0': rcnn_cfg.learning_rate,
                'lrf': None,
                'weight_decay': rcnn_cfg.weight_decay,
                'momentum': rcnn_cfg.momentum,
                'optimizer': rcnn_cfg.optimizer,
                'scheduler': 'StepLR',
                'ema': None,
                'n_augmentations': None,
                'op_iou_thresh': eval_cfg.iou_thresh,
                'fp_rate_cap': eval_cfg.fp_rate_cap,
                'max_det': eval_cfg.max_det,
                'edge_touch_k': eval_cfg.edge_touch_k,
                'conf_sweep_desc': conf_desc,
                'seed': self.config.seed,
            }
            paths = {
                'weights_best_path': str(weights_path),
                'tb_dir': str(self.output_dir / 'tensorboard'),
                'summary_json': str(self.output_dir / 'op_select_summary.json'),
            }
            # include epoch_best if we can infer; RCNN final eval lacks epoch tracking → leave blank unless meta had it
            if folds_metrics and best_meta_path.exists():
                try:
                    with open(best_meta_path, 'r') as f:
                        meta = json.load(f)
                    ep = meta.get('metrics', {}).get('epoch') or meta.get('epoch')
                    if ep is not None:
                        folds_metrics[0]['epoch_best'] = float(ep)
                        aggregate['epoch_best_mean'] = float(ep)
                except Exception:
                    pass

            time_elapsed = time.time() - _time_start
            # Single-split RCNN → mark as fold1 for compact upsert
            RunLogger.append_compact(run_info, folds_metrics, aggregate, time_elapsed, fold_index=1)
            RunLogger.append_extended(run_info, aggregate, paths)
        except Exception as _e:
            print(f"⚠️  Run logging failed: {_e}")

        print(f"✅ Training complete! Weights saved to: {weights_path}")
        return str(weights_path)

    def _train_epoch(self, data_loader, device) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            # Enable/disable autocast based on config.rcnn.amp
            if bool(getattr(self.config.rcnn, 'amp', False)) and torch.cuda.is_available():
                scaler = torch.amp.GradScaler('cuda', enabled=True)
                with torch.amp.autocast('cuda', enabled=True):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                scaler.scale(losses).backward()
                # Optional gradient clipping
                if getattr(self.config.rcnn, 'grad_clip_norm', None):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.rcnn.grad_clip_norm))
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                if getattr(self.config.rcnn, 'grad_clip_norm', None):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.rcnn.grad_clip_norm))
                self.optimizer.step()
            # unify losses for logging when amp path used above
            
            total_loss += losses.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(data_loader)}, Loss: {losses.item():.4f}")
        
        return {'loss': total_loss / num_batches}

    def _validate_epoch(self, data_loader, device) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        all_detections = []
        all_ground_truth = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                # Move to device
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                predictions = self.model(images)
                
                # Convert predictions to detection format
                for pred in predictions:
                    boxes = pred['boxes'].cpu().numpy().tolist()
                    scores = pred['scores'].cpu().numpy().tolist()
                    labels = pred['labels'].cpu().numpy().tolist()
                    
                    # Filter by confidence
                    filtered_boxes = []
                    filtered_scores = []
                    filtered_labels = []
                    
                    for box, score, label in zip(boxes, scores, labels):
                        conf_thresh = self.config.rcnn.conf or 0.5  # Fallback to 0.5 if no threshold set
                        if score >= conf_thresh:
                            filtered_boxes.append(box)
                            filtered_scores.append(score)
                            filtered_labels.append(label)
                    
                    # Store detection results (removed DetectionResult dependency)
                
                # Convert targets to ground truth format
                for target in targets:
                    boxes = target['boxes'].cpu().numpy().tolist()
                    labels = target['labels'].cpu().numpy().tolist()
                    
                    # Store ground truth (removed GroundTruth dependency)
        
        # Return simple placeholder metrics (detailed validation now done via eval.validate)
        return {
            'ap_50': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    def _save_model(self, filename: str):
        """Save model weights."""
        model_path = self.output_dir / filename
        torch.save(self.model.state_dict(), model_path)
    
    def _save_training_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _run_unified_evaluation(self, data_loader, epoch, device):
        """Run unified operating point selection evaluation."""
        from src.eval.op_select import build_val_cache, select_operating_point, save_operating_point_results, save_best_metadata
        
        print(f"\nRunning unified evaluation (epoch {epoch})...")
        
        # Get evaluation config - require it to be present
        if not hasattr(self.config, 'eval') or self.config.eval is None:
            raise ValueError("❌ eval configuration is required but not found in config.yaml")
        
        eval_config = self.config.eval
        conf_sweep = eval_config.conf_sweep
        iou_thresh = eval_config.iou_thresh
        max_det = eval_config.max_det
        edge_touch_k = eval_config.edge_touch_k
        metric = 'f1'
        
        # Convert data_loader to format expected by op_select
        val_paths = []
        labels_dict = {}
        
        for batch in data_loader:
            images, targets = batch
            for i, target in enumerate(targets):
                # Reconstruct image path (this is a limitation - we need path info)
                # For now, we'll skip unified evaluation for RCNN during training
                # and only run it at the end when we have the full dataset
                pass
        
        print("⚠️  Unified evaluation during training not yet implemented for RCNN")
        print("    Will run full evaluation after training completes")
    
    def _final_evaluation(self, data_loader, device):
        """Run final unified evaluation."""
        from src.eval.op_select import build_val_cache, select_operating_point, save_operating_point_results, save_best_metadata
        from src.datasets.canonical import load_canonical_dataset
        
        print("\nRunning final unified evaluation...")
        
        # Get evaluation config - require it to be present
        if not hasattr(self.config, 'eval') or self.config.eval is None:
            raise ValueError("❌ eval configuration is required but not found in config.yaml")
        
        eval_config = self.config.eval
        conf_sweep = eval_config.conf_sweep
        iou_thresh = eval_config.iou_thresh
        max_det = eval_config.max_det
        edge_touch_k = eval_config.edge_touch_k
        metric = 'f1'
        
        # Load validation dataset to get image paths
        try:
            dataset = load_canonical_dataset(
                labels_json=self.config.data.labels_json,
                splits_dir=self.config.data.splits_dir
            )
            
            val_paths = [sample.image_path for sample in dataset.val]
            labels_dict = {}
            for sample in dataset.val:
                labels_dict[sample.image_path] = {
                    'bboxes': [{'xyxy': box} for box in sample.boxes]
                }
            
            # Build validation cache
            pred_cache = build_val_cache(
                model=self.model,
                val_paths=val_paths,
                labels_dict=labels_dict,
                imgsz=self.config.rcnn.imgsz,
                base_conf=0.01
            )
            
            # Select operating point
            op_point = select_operating_point(
                pred_cache=pred_cache,
                labels_dict=labels_dict,
                conf_sweep=conf_sweep,
                iou_thresh=iou_thresh,
                max_det=max_det,
                edge_touch_k=edge_touch_k,
                metric=metric
            )
            
            # Update config with best threshold
            self.config.rcnn.conf = op_point.best_threshold
            print(f"✅ Selected confidence threshold: {op_point.best_threshold:.3f}")
            
            # Save results
            save_operating_point_results(op_point, self.output_dir)
            
            # Generate validation visualizations
            from src.eval.viz import save_eval_viz_from_cache
            viz_dir = self.output_dir / "val_viz"
            save_eval_viz_from_cache(pred_cache, labels_dict, viz_dir, class_name="probe")
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('val/best_conf', op_point.best_threshold, self.config.rcnn.epochs)
                self.writer.add_scalar('val/f1@best', op_point.best_metrics.f1, self.config.rcnn.epochs)
                self.writer.add_scalar('val/precision@best', op_point.best_metrics.precision, self.config.rcnn.epochs)
                self.writer.add_scalar('val/recall@best', op_point.best_metrics.recall, self.config.rcnn.epochs)
                self.writer.add_scalar('val/mean_iou@best', op_point.best_metrics.mean_iou, self.config.rcnn.epochs)
            
            # Save metadata alongside weights
            weights_path = self.output_dir / "best_model.pth"
            if weights_path.exists():
                additional_config = {
                    "nms_iou": self.config.rcnn.iou,
                    "max_det": max_det,
                    "edge_touch_k": edge_touch_k,
                    "model_type": "rcnn"
                }
                save_best_metadata(op_point, weights_path, additional_config)
            
        except Exception as e:
            print(f"⚠️  Error in unified evaluation: {e}")
            print("    Falling back to standard evaluation")
        
        # Load best model
        best_model_path = self.output_dir / "best_model.pth"
        self.model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        # Run evaluation
        val_metrics = self._validate_epoch(data_loader, device)
        
        # Print results
        print(f"RCNN Final Evaluation Results: {val_metrics}")
    
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
        
        conf = conf or self.config.rcnn.conf or 0.5  # Fallback to 0.5 if no threshold set
        
        # Load and preprocess image
        from PIL import Image
        import torchvision.transforms as transforms
        
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size
        transform = transforms.Compose([
            transforms.Resize((self.config.rcnn.imgsz, self.config.rcnn.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # Move to device
        device = self._resolve_device()
        image_tensor = image_tensor.to(device)
        
        # Run inference
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Filter by confidence
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy().tolist()
        scores = pred['scores'].cpu().numpy().tolist()
        labels = pred['labels'].cpu().numpy().tolist()
        
        # Optional edge-touch filter
        try:
            k = getattr(self.config, 'eval').edge_touch_k
        except Exception:
            k = None
        if k is not None:
            keep = edge_touch_filter(boxes, orig_h, orig_w, k)
            boxes = [b for b, m in zip(boxes, keep) if m]
            scores = [s for s, m in zip(scores, keep) if m]
            labels = [l for l, m in zip(labels, keep) if m]

        # Confidence filter
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= conf:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label)

        # Enforce max_det by keeping top-k
        try:
            max_det = int(getattr(self.config, 'eval').max_det)
        except Exception:
            max_det = 1
        if len(filtered_scores) > max_det:
            order = sorted(range(len(filtered_scores)), key=lambda i: filtered_scores[i], reverse=True)[:max_det]
            filtered_boxes = [filtered_boxes[i] for i in order]
            filtered_scores = [filtered_scores[i] for i in order]
            filtered_labels = [filtered_labels[i] for i in order]
        
        return {
            'boxes': filtered_boxes,
            'scores': filtered_scores,
            'labels': filtered_labels
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {}
        
        return {
            'model_name': self.config.model,
            'weights_path': str(self.output_dir / "best_model.pth"),
            'input_size': self.config.rcnn.imgsz,
            'device': self.config.device,
            'run_name': self.run_name,
            'output_dir': str(self.output_dir)
        }
