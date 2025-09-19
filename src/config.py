"""
Unified configuration management for probe detection training.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data configuration."""
    images_root: str
    labels_json: str
    splits_dir: str
    class_names: List[str]


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""
    # Geometry
    degrees: float = 2.0
    translate: float = 0.05
    scale_min: float = 0.9
    scale_max: float = 1.1
    shear: float = 2.0
    fliplr_p: float = 0.5
    flipud_p: float = 0.0
    
    # Photometric
    brightness: float = 0.2
    contrast: float = 0.2
    brightness_contrast_p: float = 0.5
    gamma: List[float] = None
    gamma_p: float = 0.3
    motion_blur_p: float = 0.25
    motion_blur_limit: List[int] = None  # e.g., [3, 9]
    noise_p: float = 0.2
    noise_std_range: List[float] = None  # e.g., [0.01, 0.05]
    clahe_p: float = 0.2
    clahe_clip_limit: float = 4.0
    clahe_tile_grid_size: List[int] = None  # e.g., [8, 8]
    glare_p: float = 0.2
    glare_flare_roi: List[float] = None  # e.g., [0, 0, 1, 0.5]
    glare_src_radius: int = 100
    glare_src_color: List[int] = None  # e.g., [255, 255, 255]
    glare_angle_range: List[float] = None  # e.g., [0, 1]
    glare_num_circles_range: List[int] = None  # e.g., [3, 6]
    glare_method: str = 'overlay'
    vignette_p: float = 0.2  # implemented via RandomShadow
    vignette_shadow_roi: List[float] = None  # e.g., [0, 0.5, 1, 1]
    vignette_num_shadows_limit: List[int] = None  # e.g., [1, 2]
    vignette_shadow_dimension: int = 5
    vignette_intensity_range: List[float] = None  # e.g., [0.3, 0.7]
    
    # Padding
    use_reflect: bool = False
    pixels: int = 0
    
    
    def __post_init__(self):
        if self.gamma is None:
            self.gamma = [0.8, 1.2]
        if self.motion_blur_limit is None:
            self.motion_blur_limit = [3, 9]
        if self.noise_std_range is None:
            self.noise_std_range = [0.01, 0.05]
        if self.clahe_tile_grid_size is None:
            self.clahe_tile_grid_size = [8, 8]
        if self.glare_flare_roi is None:
            self.glare_flare_roi = [0, 0, 1, 0.5]
        if self.glare_src_color is None:
            self.glare_src_color = [255, 255, 255]
        if self.glare_angle_range is None:
            self.glare_angle_range = [0, 1]
        if self.glare_num_circles_range is None:
            self.glare_num_circles_range = [3, 6]
        if self.vignette_shadow_roi is None:
            self.vignette_shadow_roi = [0, 0.5, 1, 1]
        if self.vignette_num_shadows_limit is None:
            self.vignette_num_shadows_limit = [1, 2]
        if self.vignette_intensity_range is None:
            self.vignette_intensity_range = [0.3, 0.7]


@dataclass
class YOLOConfig:
    """YOLO-specific training configuration."""
    weights: str = "yolov8n.pt"
    imgsz: int = 640
    epochs: int = 80
    batch_size: int = 32
    conf: float = None  # Will be determined dynamically during validation
    iou: float = 0.5
    rect: bool = True
    patience: int = 20
    ema: bool = True
    n_augmentations: int = 1  # Number of augmented versions per image
    # Optimizer hyperparams
    optimizer: str = "auto"  # 'auto' | 'SGD' | 'Adam' | 'AdamW' | 'NAdam' | 'RMSProp'
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    lrf: float = 0.01
    cos_lr: bool = False
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.001



@dataclass
class RCNNConfig:
    """Faster R-CNN specific training configuration."""
    weights: str = "pretrained"
    imgsz: int = 800
    epochs: int = 50
    batch_size: int = 4
    conf: float = None  # Will be determined dynamically during validation
    iou: float = 0.5
    backbone: str = "resnet50_fpn"
    learning_rate: float = 0.005
    weight_decay: float = 0.0005
    optimizer: str = "AdamW"  # 'SGD' | 'Adam' | 'AdamW' | 'RMSprop'
    momentum: float = 0.9  # used by SGD/RMSprop
    patience: int = 20
    # Performance/optimization
    amp: bool = False
    grad_clip_norm: float | None = None
    # RPN configuration
    rpn_pre_nms_topk_train: int | None = None
    rpn_pre_nms_topk_test: int | None = None
    rpn_post_nms_topk_train: int | None = None
    rpn_post_nms_topk_test: int | None = None
    rpn_nms_thresh: float | None = None
    rpn_fg_iou_thresh: float | None = None
    rpn_bg_iou_thresh: float | None = None
    # Anchors
    anchor_sizes: List[int] | None = None
    anchor_aspect_ratios: List[float] | None = None
    # ROI heads / box settings
    box_score_thresh: float | None = None
    box_nms_thresh: float | None = None
    box_fg_iou_thresh: float | None = None
    box_bg_iou_thresh: float | None = None
    detections_per_img: int | None = None


@dataclass
class EvalConfig:
    """Unified evaluation configuration."""
    conf_sweep: List[float] = None
    iou_thresh: float = 0.8
    max_det: int = 1
    edge_touch_k: int = 0  # 0 = no edge constraint
    fp_rate_cap: float = 0.05
    save_sweep_tables: bool = False
    save_val_viz: bool = False
    final_viz: bool = True
    # Early stopping control (single policy)
    early_stopping_metric: str = "f1_best"  # 'f1_best' only
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.002  # use 0.001 if metric=='map'
    es_min_epoch: int = None
    
    def __post_init__(self):
        if self.conf_sweep is None:
            # Default confidence sweep: 0.01 to 0.20 in 0.01 steps for probe detection
            self.conf_sweep = [round(0.01 + i * 0.01, 2) for i in range(20)]  # 0.01, 0.02, ..., 0.20
        # No-op for removed fields


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig
    aug: AugmentationConfig
    yolo: YOLOConfig
    rcnn: RCNNConfig
    eval: EvalConfig
    seed: int = 42
    device: str = "mps"  # "mps" for Apple Silicon, "cpu" for CPU, "0" for CUDA GPU
    model: str = "yolo_nano"  # "yolo_v8", "yolo_nano", "faster_rcnn"
    name: Optional[str] = None
    save_dir: Optional[str] = None


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create nested config objects
    # Evaluation config
    eval_cfg_dict = dict(config_dict.get('eval', {}) or {})
    config = Config(
        seed=config_dict.get('seed', 42),
        device=config_dict.get('device', 'mps'),
        model=config_dict.get('model', 'yolo_nano'),
        data=DataConfig(**config_dict.get('data', {})),
        aug=AugmentationConfig(**config_dict.get('aug', {})),
        yolo=YOLOConfig(**config_dict.get('yolo', {})),
        rcnn=RCNNConfig(**config_dict.get('rcnn', {})),
        eval=EvalConfig(**eval_cfg_dict),
        name=config_dict.get('name'),
        save_dir=config_dict.get('save_dir')
    )
    
    return config


def save_config(config: Config, path: str) -> None:
    """Save configuration to YAML file."""
    config_dict = {
        'seed': config.seed,
        'device': config.device,
        'model': config.model,
        'data': {
            'images_root': config.data.images_root,
            'labels_json': config.data.labels_json,
            'splits_dir': config.data.splits_dir,
            'class_names': config.data.class_names
        },
        'aug': {
            'degrees': config.aug.degrees,
            'translate': config.aug.translate,
            'scale_min': config.aug.scale_min,
            'scale_max': config.aug.scale_max,
            'shear': config.aug.shear,
            'fliplr_p': config.aug.fliplr_p,
            'flipud_p': config.aug.flipud_p,
            'brightness': config.aug.brightness,
            'contrast': config.aug.contrast,
            'brightness_contrast_p': config.aug.brightness_contrast_p,
            'gamma': config.aug.gamma,
            'gamma_p': config.aug.gamma_p,
            'motion_blur_p': config.aug.motion_blur_p,
            'motion_blur_limit': config.aug.motion_blur_limit,
            'noise_p': config.aug.noise_p,
            'noise_std_range': config.aug.noise_std_range,
            'clahe_p': config.aug.clahe_p,
            'clahe_clip_limit': config.aug.clahe_clip_limit,
            'clahe_tile_grid_size': config.aug.clahe_tile_grid_size,
            'glare_p': config.aug.glare_p,
            'glare_flare_roi': config.aug.glare_flare_roi,
            'glare_src_radius': config.aug.glare_src_radius,
            'glare_src_color': config.aug.glare_src_color,
            'glare_angle_range': config.aug.glare_angle_range,
            'glare_num_circles_range': config.aug.glare_num_circles_range,
            'glare_method': config.aug.glare_method,
            'vignette_p': config.aug.vignette_p,
            'vignette_shadow_roi': config.aug.vignette_shadow_roi,
            'vignette_num_shadows_limit': config.aug.vignette_num_shadows_limit,
            'vignette_shadow_dimension': config.aug.vignette_shadow_dimension,
            'vignette_intensity_range': config.aug.vignette_intensity_range,
            'use_reflect': config.aug.use_reflect,
            'pixels': config.aug.pixels,
        },
        'yolo': {
            'weights': config.yolo.weights,
            'imgsz': config.yolo.imgsz,
            'epochs': config.yolo.epochs,
            'batch_size': config.yolo.batch_size,
            'conf': config.yolo.conf,
            'iou': config.yolo.iou,
            'rect': config.yolo.rect,
            'patience': config.yolo.patience,
            'ema': config.yolo.ema,
            'n_augmentations': config.yolo.n_augmentations,
            'optimizer': config.yolo.optimizer,
            'lr0': config.yolo.lr0,
            'momentum': config.yolo.momentum,
            'weight_decay': config.yolo.weight_decay,
            'lrf': config.yolo.lrf,
            'cos_lr': config.yolo.cos_lr,
            'warmup_epochs': config.yolo.warmup_epochs,
            'warmup_momentum': config.yolo.warmup_momentum,
            'warmup_bias_lr': config.yolo.warmup_bias_lr
        },
        'rcnn': {
            'weights': config.rcnn.weights,
            'imgsz': config.rcnn.imgsz,
            'epochs': config.rcnn.epochs,
            'batch_size': config.rcnn.batch_size,
            'conf': config.rcnn.conf,
            'iou': config.rcnn.iou,
            'backbone': config.rcnn.backbone,
            'learning_rate': config.rcnn.learning_rate,
            'weight_decay': config.rcnn.weight_decay,
            'optimizer': config.rcnn.optimizer,
            'momentum': config.rcnn.momentum,
            'patience': config.rcnn.patience,
            'amp': config.rcnn.amp,
            'grad_clip_norm': config.rcnn.grad_clip_norm,
            'rpn_pre_nms_topk_train': config.rcnn.rpn_pre_nms_topk_train,
            'rpn_pre_nms_topk_test': config.rcnn.rpn_pre_nms_topk_test,
            'rpn_post_nms_topk_train': config.rcnn.rpn_post_nms_topk_train,
            'rpn_post_nms_topk_test': config.rcnn.rpn_post_nms_topk_test,
            'rpn_nms_thresh': config.rcnn.rpn_nms_thresh,
            'rpn_fg_iou_thresh': config.rcnn.rpn_fg_iou_thresh,
            'rpn_bg_iou_thresh': config.rcnn.rpn_bg_iou_thresh,
            'anchor_sizes': config.rcnn.anchor_sizes,
            'anchor_aspect_ratios': config.rcnn.anchor_aspect_ratios,
            'box_score_thresh': config.rcnn.box_score_thresh,
            'box_nms_thresh': config.rcnn.box_nms_thresh,
            'box_fg_iou_thresh': config.rcnn.box_fg_iou_thresh,
            'box_bg_iou_thresh': config.rcnn.box_bg_iou_thresh,
            'detections_per_img': config.rcnn.detections_per_img,
        },
        'eval': {
            'iou_thresh': config.eval.iou_thresh,
            'conf_sweep': config.eval.conf_sweep,
            'max_det': config.eval.max_det,
            'edge_touch_k': config.eval.edge_touch_k,
            'fp_rate_cap': config.eval.fp_rate_cap,
            'save_sweep_tables': config.eval.save_sweep_tables,
            'save_val_viz': config.eval.save_val_viz,
            'early_stopping_metric': config.eval.early_stopping_metric,
            'early_stopping_patience': config.eval.early_stopping_patience,
            'early_stopping_min_delta': config.eval.early_stopping_min_delta,
            'es_min_epoch': config.eval.es_min_epoch,
        },
        'name': config.name,
        'save_dir': config.save_dir
    }
    
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
