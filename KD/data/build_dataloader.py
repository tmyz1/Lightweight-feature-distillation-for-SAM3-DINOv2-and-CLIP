from __future__ import annotations

import sys
import random
import secrets
from functools import partial
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
from sam3.train.transforms.basic import (
    get_random_resize_max_size,
    get_random_resize_scales,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    PadToSizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    pad,
)
from sam3.train.transforms.filter_query_transforms import (
    FilterCrowds,
    FilterEmptyTargets,
    FilterFindQueriesWithTooManyOut,
    FlexibleFilterFindGetQueries,
)
from sam3.train.transforms.point_sampling import RandomizeInputBbox
from sam3.train.transforms.segmentation import DecodeRle

from KD.data.SA_1B import build_sa1b_dataset, is_sa1b_dataset


def get_split_root(dataset_cfg: Dict[str, Any], training: bool) -> Path:
    """Use an optional independent root for COCO validation data."""
    root_key = "train_root" if training else "val_root"
    return Path(dataset_cfg.get(root_key, dataset_cfg["root"]))


def resize_img_batch(img_batch: torch.Tensor, resolution: int) -> torch.Tensor:
    if img_batch.shape[-2:] == (resolution, resolution):
        return img_batch
    return F.interpolate(
        img_batch,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
    )


class PadToSizeWithValidRegionAPI(PadToSizeAPI):
    """Pad images while retaining the pre-padding content box for KD losses."""

    def __call__(self, datapoint, **kwargs):
        for index, image in enumerate(datapoint.images):
            width, height = image.data.size
            if self.bottom_right:
                padding = (self.size - width, self.size - height)
                left = top = 0
            else:
                padding = self._sample_pad(width, height)
                left, top = padding[:2]

            image.kd_valid_box = torch.tensor(
                [left, top, left + width, top + height], dtype=torch.float32
            )
            datapoint = pad(datapoint, index, padding, v2=self.v2)
        return datapoint


#针对不同的模型采用不同的分辨率
def add_multi_resolution_batches(batch, cfg: Dict[str, Any]):
    dataset_cfg = cfg["dataset"]
    sam3_resolution = int(dataset_cfg.get("resolution", 1008))
    student_resolution = int(
        cfg.get("Student", {}).get("kd_resolution", sam3_resolution)
    )
    dino_v2_resolution = int(cfg.get("DINO_V2", {}).get("resolution", 224))
    clip_resolution = int(cfg.get("CLIP", {}).get("resolution", 224))

    batch.sam3_img_batch = resize_img_batch(batch.img_batch, sam3_resolution)
    batch.student_img_batch = resize_img_batch(batch.img_batch, student_resolution)
    batch.dino_v2_img_batch = resize_img_batch(batch.img_batch, dino_v2_resolution)
    batch.clip_img_batch = resize_img_batch(batch.img_batch, clip_resolution)
    return batch


def collate_kd_batch(batch, cfg: Dict[str, Any], enable_segmentation: bool):
    valid_boxes = []
    for datapoint in batch:
        for image in datapoint.images:
            valid_box = getattr(image, "kd_valid_box", None)
            if valid_box is None:
                height, width = image.size
                valid_box = torch.tensor(
                    [0, 0, width, height], dtype=torch.float32
                )
            valid_boxes.append(valid_box)

    batched = collate_fn_api(
        batch,
        dict_key="all",
        with_seg_masks=enable_segmentation,
        repeats=1,
    )["all"]
    canvas_height, canvas_width = batched.img_batch.shape[-2:]
    canvas_scale = torch.tensor(
        [canvas_width, canvas_height, canvas_width, canvas_height],
        dtype=torch.float32,
    )
    batched.kd_valid_boxes = torch.stack(valid_boxes) / canvas_scale
    return add_multi_resolution_batches(batched, cfg)


def build_transforms(resolution: int, max_ann_per_img: int = 200):
    return [
        ComposeAPI(
            transforms=[
                FlexibleFilterFindGetQueries(query_filter=FilterCrowds()),
                DecodeRle(),
                RandomizeInputBbox(box_noise_std=0.1, box_noise_max=20),
                RandomResizeAPI(
                    sizes=get_random_resize_scales(
                        size=resolution,
                        min_size=480,
                        rounded=False,
                    ),
                    max_size=get_random_resize_max_size(size=resolution),
                    square=True,
                    consistent_transform=False,
                ),
                PadToSizeWithValidRegionAPI(
                    size=resolution, consistent_transform=False
                ),
                ToTensorAPI(),
                FlexibleFilterFindGetQueries(query_filter=FilterEmptyTargets()),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                FlexibleFilterFindGetQueries(query_filter=FilterEmptyTargets()),
            ]
        ),
        FlexibleFilterFindGetQueries(
            query_filter=FilterFindQueriesWithTooManyOut(
                max_num_objects=max_ann_per_img
            )
        ),
    ]


def build_val_transforms(resolution: int):
    return [
        ComposeAPI(
            transforms=[
                DecodeRle(),
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=get_random_resize_max_size(size=resolution),
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    ]


def resolve_split_paths(root: Path, split: str) -> tuple[Path, Path]:
    image_candidates = (
        root / split,
        root / "train" / split,
        root / "val" / split,
        root / "images" / split,
    )
    image_folder = next(
        (candidate for candidate in image_candidates if candidate.is_dir()), None
    )
    if image_folder is None:
        candidates = ", ".join(str(candidate) for candidate in image_candidates)
        raise FileNotFoundError(
            f"Cannot find image folder for split={split!r}. Checked: {candidates}"
        )

    local_instance_annotations = tuple(sorted(image_folder.glob("instances_*.json")))
    local_json_annotations = tuple(sorted(image_folder.glob("*.json")))
    unique_local_json = (
        local_json_annotations if len(local_json_annotations) == 1 else ()
    )
    annotation_candidates = (
        image_folder / "_annotations.coco.json",
        image_folder / f"instances_{split}.json",
        *local_instance_annotations,
        *unique_local_json,
        root / "annotations" / f"instances_{split}.json",
        root / "annotations_trainval2017" / "annotations" / f"instances_{split}.json",
    )
    ann_file = next(
        (candidate for candidate in annotation_candidates if candidate.is_file()), None
    )
    if ann_file is None:
        candidates = ", ".join(str(candidate) for candidate in annotation_candidates)
        raise FileNotFoundError(
            f"Cannot find COCO annotations for split={split!r}. Checked: {candidates}"
        )
    return image_folder, ann_file


def limit_dataset_to_images(
    dataset: Sam3ImageDataset,
    num_images: int,
    split_name: str,
    subset_seed: int,
) -> None:
    """Limit a dataset by source images while retaining every category chunk."""
    if num_images < 1:
        raise ValueError(
            f"dataset.{split_name}_num_images must be at least 1 when specified."
        )

    num_source_images = len(dataset.coco._raw_data)
    num_category_chunks = len(dataset.coco.category_chunks)
    selected_count = min(num_images, num_source_images)
    source_indices = list(range(num_source_images))
    random.Random(subset_seed).shuffle(source_indices)
    source_indices = sorted(source_indices[:selected_count])

    chunk_offsets = torch.arange(num_category_chunks, dtype=torch.long)
    selected_indices = torch.as_tensor(source_indices, dtype=torch.long)
    dataset.ids = (
        selected_indices[:, None] * num_category_chunks + chunk_offsets[None, :]
    ).reshape(-1)
    dataset.repeat_factors = torch.ones(len(dataset.ids), dtype=torch.float32)
    print(
        f"{split_name.capitalize()} dataset limited to "
        f"{selected_count} source image(s), {len(dataset.ids)} datapoint(s), "
        f"subset_seed={subset_seed}."
    )


def get_subset_seed(
    dataset_cfg: Dict[str, Any],
    split_name: str,
    default_seed: int | None = None,
) -> int:
    """Use an optional configured seed, fixed default, or fresh random seed."""
    configured_seed = dataset_cfg.get(f"{split_name}_subset_seed")
    if configured_seed is not None:
        return int(configured_seed)
    if default_seed is not None:
        return int(default_seed)
    return secrets.randbits(63)


def get_selected_coco_image_ids(dataset: Sam3ImageDataset) -> list[int]:
    """Return original COCO image ids represented by the current dataset ids."""
    num_category_chunks = len(dataset.coco.category_chunks)
    source_indices = torch.unique(dataset.ids // num_category_chunks).tolist()
    return sorted(
        int(dataset.coco._raw_data[index]["image"]["id"])
        for index in source_indices
    )


def build_split_dataloader(
    cfg: Dict[str, Any],
    split: str,
    training: bool,
    limit_ids=None,
    shuffle: bool = True,
    drop_last: bool = True,
    distributed: bool | None = None,
) -> DataLoader:
    dataset_cfg = cfg["dataset"]
    root = get_split_root(dataset_cfg, training)
    resolution = int(dataset_cfg.get("resolution", 1008))
    enable_segmentation = bool(dataset_cfg.get("enable_segmentation", False))
    eval_category_chunk_size = int(dataset_cfg.get("eval_category_chunk_size", 2))
    max_objects_per_query = int(dataset_cfg.get("max_objects_per_query", 200))
    if eval_category_chunk_size < 1:
        raise ValueError("dataset.eval_category_chunk_size must be at least 1.")
    if max_objects_per_query < 1:
        raise ValueError("dataset.max_objects_per_query must be at least 1.")

    transforms = (
        build_transforms(resolution, max_ann_per_img=max_objects_per_query)
        if training
        else build_val_transforms(resolution)
    )
    split_name = "train" if training else "val"
    if is_sa1b_dataset(dataset_cfg, training=training):
        subset_seed = get_subset_seed(
            dataset_cfg,
            split_name,
            default_seed=None if training else 0,
        )
        shard_key = "sa1b_train_shards" if training else "sa1b_val_shards"
        dataset = build_sa1b_dataset(
            root=root,
            transforms=transforms,
            load_segmentation=enable_segmentation,
            training=training,
            max_ann_per_img=500000,
            max_images=None if limit_ids is None else int(limit_ids),
            subset_seed=subset_seed,
            shard_names=dataset_cfg.get(shard_key, dataset_cfg.get("sa1b_shards")),
        )
    else:
        img_folder, ann_file = resolve_split_paths(root, split)
        dataset = Sam3ImageDataset(
            img_folder=str(img_folder),
            ann_file=str(ann_file),
            transforms=transforms,
            load_segmentation=enable_segmentation,
            max_ann_per_img=500000,
            multiplier=1,
            max_train_queries=50000,
            max_val_queries=50000,
            training=training,
            use_caching=False,
            coco_json_loader=(
                COCO_FROM_JSON
                if training
                else partial(
                    COCO_FROM_JSON,
                    include_negatives=True,
                    category_chunk_size=eval_category_chunk_size,
                )
            ),
            limit_ids=None,
        )
        if limit_ids is not None:
            subset_seed = get_subset_seed(
                dataset_cfg,
                split_name,
                default_seed=None if training else len(dataset.coco._raw_data),
            )
            limit_dataset_to_images(
                dataset,
                int(limit_ids),
                split_name,
                subset_seed,
            )

    if distributed is None:
        distributed = bool(cfg.get("multi_gpu", {}).get("enabled", False))
    use_distributed_sampler = bool(distributed) and dist.is_available() and dist.is_initialized()
    sampler = None
    if use_distributed_sampler:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )

    return DataLoader(
        dataset,
        batch_size=int(dataset_cfg.get("batch_size", 1)),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=lambda batch: collate_kd_batch(batch, cfg, enable_segmentation),
    )


def build_dataloader(cfg: Dict[str, Any]) -> DataLoader:
    dataset_cfg = cfg["dataset"]
    return build_split_dataloader(
        cfg,
        split=dataset_cfg.get("train_split", "train"),
        training=True,
        limit_ids=dataset_cfg.get("train_num_images", 1000),
        shuffle=True,
        drop_last=True,
        distributed=True,
    )


def build_val_dataloader(cfg: Dict[str, Any]) -> DataLoader:
    dataset_cfg = cfg["dataset"]
    return build_split_dataloader(
        cfg,
        split=dataset_cfg.get("val_split", "valid"),
        training=False,
        limit_ids=dataset_cfg.get("val_num_images"),
        shuffle=False,
        drop_last=False,
        distributed=False,
    )
