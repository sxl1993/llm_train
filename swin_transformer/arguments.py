from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class FeatureExtractionArguments:
    do_normalize: bool = field(
        default=True,
        metadata={"help": "Whether to apply normalization to the images."}
    )
    do_rescale: bool = field(
        default=True,
        metadata={"help": "Whether to rescale the pixel values of the images."}
    )
    do_resize: bool = field(
        default=True,
        metadata={"help": "Whether to resize the images to a specific size."}
    )
    image_mean: List[float] = field(
        default_factory=lambda: None,
        metadata={"help": "The mean values to use for normalizing the images."}
    )
    image_std: List[float] = field(
        default_factory=lambda: None,
        metadata={"help": "The standard deviation values to use for normalizing the images."}
    )
    resample: int = field(
        default=3,
        metadata={"help": "The resampling filter to use when resizing images (0: NEAREST, 1: BILINEAR, 2: BICUBIC, 3: LANCZOS)."}
    )
    size: int = field(
        default=224,
        metadata={"help": "The target size (height and width) to resize the images to."}
    )


@dataclass
class DatasetArguments:
    data_dir: str = field(
        default=None,
        metadata={"help": "The directory where the dataset is stored."}
    )

@dataclass
class ModelArguments:
    embed_dim: int = field(
        default=128,
        metadata={"help": "The embedding dimension of the model."}
    )
    depths: List[int] = field(
        default_factory=lambda: None,
        metadata={"help": "A list of integers representing the depth of each layer."}
    )
    num_heads: List[int] = field(
        default_factory=lambda: None,
        metadata={"help": "A list of integers representing the number of attention heads in each layer."}
    )
    path_norm: bool = field(
        default=True,
        metadata={"help": "Whether to apply path normalization to the model."}
    )
    num_labels: int = field(
        default=None,
        metadata={"help": "The number of labels in the classification task."}
    )
    id2label: Dict[int, str] = field(
        default_factory=dict,
        metadata={"help": "A dictionary mapping IDs to labels."}
    )
    label2id: Dict[str, int] = field(
        default_factory=dict,
        metadata={"help": "A dictionary mapping labels to IDs."}
    )
    ignore_mismatched_sizes: bool = field(
        default=True,
        metadata={"help": "Whether to ignore mismatched sizes between the model and the checkpoint."}
    )

# 假设 labels 是从数据集中获取的类别名称列表
# labels = ds['train'].features["label"].names
# model_cfg = ModelConfig(
#     depths=[2, 2, 18, 2],
#     num_heads=[4, 8, 16, 32],
#     num_labels=len(labels),
#     id2label={str(i): c for i, c in enumerate(labels)},
#     label2id={c: str(i) for i, c in enumerate(labels)}
# )

# 初始化配置
# dataset_cfg = DatasetArguments()
# # 初始化配置
# feat_ext_cfg = FeatureExtractionConfig(
#     image_mean=[0.485, 0.456, 0.406],
#     image_std=[0.229, 0.224, 0.225],
#     size=224
# )