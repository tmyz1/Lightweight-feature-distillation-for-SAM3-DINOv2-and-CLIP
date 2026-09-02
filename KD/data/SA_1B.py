from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch

from sam3.train.data.coco_json_loaders import (
    ann_to_rle,
    convert_boxlist_to_normalized_tensor,
)
from sam3.train.data.sam3_image_dataset import Sam3ImageDataset


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SA1B_DATASET_TYPES = {"sa1b", "sa-1b", "sa_1b"}
OBJECT_CATEGORY_ID = 1
OBJECT_QUERY_TEXT = "object"


@dataclass(frozen=True)
class SA1BRecord:
    """One SA-1B image and its corresponding per-image annotation JSON."""

    image_id: int
    image_path: str
    annotation_path: str


def is_sa1b_dataset(
    dataset_cfg: Dict[str, Any],
    training: bool | None = None,
) -> bool:
    """Return whether the selected train or validation split uses direct SA-1B loading."""
    split_type_key = "train_type" if training else "val_type"
    dataset_type = dataset_cfg.get(
        split_type_key if training is not None else "type",
        dataset_cfg.get("type", dataset_cfg.get("format", "coco")),
    )
    return str(dataset_type).strip().lower() in SA1B_DATASET_TYPES


def _normalize_shard_names(shard_names: Any) -> set[str] | None:
    if shard_names is None:
        return None
    if isinstance(shard_names, str):
        return {shard_names}
    if isinstance(shard_names, Sequence):
        return {str(name) for name in shard_names}
    raise TypeError("SA-1B shard names must be a string, a list of strings, or null.")


"""
Find all subfolders in the root directory that meet the specified criteria.
"""
def _get_sa1b_shard_dirs(root: Path, shard_names: Any) -> list[Path]:
    requested_shards = _normalize_shard_names(shard_names)
    shard_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path.name.startswith("sa_") or path.name.startswith("images"))
        and (requested_shards is None or path.name in requested_shards)
    )
    if requested_shards is not None:
        found_shards = {path.name for path in shard_dirs}
        missing_shards = sorted(requested_shards - found_shards)
        if missing_shards:
            raise FileNotFoundError(
                "Requested SA-1B shard directories do not exist: "
                + ", ".join(missing_shards)
            )
    if not shard_dirs:
        raise FileNotFoundError(
            f"No SA-1B shard directories matching 'sa_*' were found under {root}."
        )

    return shard_dirs

"""
Within an SA-1B shard directory, locate the corresponding images and JSON annotation files.
"""
def _iter_pairs_in_shard(shard_dir: Path) -> Iterable[tuple[Path, Path]]:
    for annotation_path in shard_dir.iterdir():
        if not annotation_path.is_file() or annotation_path.suffix.lower() != ".json":
            continue
        image_path = next(
            (
                annotation_path.with_suffix(suffix)
                for suffix in IMAGE_SUFFIXES
                if annotation_path.with_suffix(suffix).is_file()
            ),
            None,
        )
        if image_path is not None:
            yield image_path, annotation_path

"""
Scan the SA-1B shard directory, find the pairs of images and JSON annotations, and organize them into a SA1BRecord list.
"""
def discover_sa1b_records(
    root: Path,
    shard_names: Any = None,
    max_images: int | None = None,
    subset_seed: int | None = None,
) -> tuple[list[SA1BRecord], int]:
    """Index every configured shard, retaining only a sampled subset when requested."""
    if max_images is not None and max_images < 1:
        raise ValueError("dataset.train_num_images must be at least 1 when specified.")

    shard_dirs = _get_sa1b_shard_dirs(root, shard_names)
    selected_paths: list[tuple[Path, Path]] = []
    if max_images is None:
        for shard_dir in shard_dirs:
            selected_paths.extend(_iter_pairs_in_shard(shard_dir))
        matched_count = len(selected_paths)
    else:
        # A full SA-1B installation may contain millions of files.  Interleave
        # shard iterators so a bounded training subset does not require a full
        # directory walk before the first batch can be produced.
        shuffled_shards = list(shard_dirs)
        random.Random(subset_seed).shuffle(shuffled_shards)
        shard_iterators = [iter(_iter_pairs_in_shard(shard_dir)) for shard_dir in shuffled_shards]
        while len(selected_paths) < max_images and shard_iterators:
            next_iterators = []
            for iterator in shard_iterators:
                try:
                    selected_paths.append(next(iterator))
                    next_iterators.append(iterator)
                    if len(selected_paths) == max_images:
                        break
                except StopIteration:
                    continue
            shard_iterators = next_iterators
        matched_count = len(selected_paths)

    if not selected_paths:
        raise RuntimeError(
            f"No matching image/JSON pairs were found in the SA-1B shards under {root}."
        )

    selected_paths.sort(key=lambda paths: str(paths[0]).lower())
    records = [
        SA1BRecord(
            image_id=index,
            image_path=image_path.relative_to(root).as_posix(),
            annotation_path=annotation_path.relative_to(root).as_posix(),
        )
        for index, (image_path, annotation_path) in enumerate(selected_paths, start=1)
    ]
    return records, matched_count


class SA1BFromIndividualJson:
    """SAM3 COCO-loader-compatible API backed by SA-1B per-image JSON files."""

    def __init__(
        self,
        annotation_file: str,
        records: Sequence[SA1BRecord],
        data_root: Path,
    ):
        del annotation_file
        self.records = tuple(records)
        self.data_root = Path(data_root)
        self._cat_idx_to_text = {OBJECT_CATEGORY_ID: OBJECT_QUERY_TEXT}
        self._sorted_cat_ids = [OBJECT_CATEGORY_ID]
        self.category_chunks = [[OBJECT_CATEGORY_ID]]
        self._raw_data = [
            {
                "image": {
                    "id": record.image_id,
                    "file_name": record.image_path,
                },
                "annotations": [],
            }
            for record in self.records
        ]

    def getDatapointIds(self) -> list[int]:
        return list(range(len(self.records)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx: int):
        record = self.records[idx]
        source_path = self.data_root / record.annotation_path
        with source_path.open("r", encoding="utf-8") as file:
            source_data = json.load(file)

        image_info = source_data.get("image", {})
        width = int(image_info["width"])
        height = int(image_info["height"])
        annotations = []
        object_ids = []
        for source_annotation in source_data.get("annotations", []):
            bbox = source_annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            annotation_id = len(annotations)
            normalized_bbox = convert_boxlist_to_normalized_tensor(
                [bbox], width, height
            )[0]
            annotation = {
                "id": annotation_id,
                "object_id": annotation_id,
                "image_id": 0,
                "bbox": normalized_bbox,
                "area": float(normalized_bbox[2] * normalized_bbox[3]),
                "is_crowd": int(source_annotation.get("iscrowd", 0)),
            }
            segmentation = source_annotation.get("segmentation")
            if segmentation:
                annotation["segmentation"] = ann_to_rle(segmentation, image_info)
            annotations.append(annotation)
            object_ids.append(annotation_id)

        queries = [
            {
                "id": 0,
                "original_cat_id": OBJECT_CATEGORY_ID,
                "object_ids_output": object_ids,
                "query_text": OBJECT_QUERY_TEXT,
                "query_processing_order": 0,
                "ptr_x_query_id": None,
                "ptr_y_query_id": None,
                "image_id": 0,
                "input_box": None,
                "input_box_label": None,
                "input_points": None,
                "is_exhaustive": True,
            }
        ]
        return queries, annotations

    def loadImagesFromDatapoint(self, idx: int):
        record = self.records[idx]
        return [
            {
                "id": 0,
                "file_name": record.image_path,
                "original_img_id": record.image_id,
                "coco_img_id": record.image_id,
            }
        ]


def build_sa1b_dataset(
    root: Path,
    transforms,
    load_segmentation: bool,
    training: bool,
    max_ann_per_img: int,
    max_images: int | None,
    subset_seed: int | None,
    shard_names: Any = None,
) -> Sam3ImageDataset:
    """Build a SAM3 dataset without first creating a monolithic COCO JSON file."""
    records, matched_count = discover_sa1b_records(
        root=root,
        shard_names=shard_names,
        max_images=max_images,
        subset_seed=subset_seed,
    )
    loader_with_records = partial(
        SA1BFromIndividualJson,
        records=records,
        data_root=root,
    )
    dataset = Sam3ImageDataset(
        img_folder=str(root),
        # Sam3ImageDataset checks that ann_file exists before invoking the loader.
        ann_file=str(root / records[0].annotation_path),
        transforms=transforms,
        load_segmentation=load_segmentation,
        max_ann_per_img=max_ann_per_img,
        multiplier=1,
        max_train_queries=50000,
        max_val_queries=50000,
        training=training,
        use_caching=False,
        coco_json_loader=loader_with_records,
        limit_ids=None,
    )
    if max_images is None:
        summary = f"indexed {matched_count} image/JSON pair(s)"
    else:
        summary = f"selected {len(records)} image/JSON pair(s) across configured shards"
    print(f"SA-1B {'train' if training else 'val'} dataset: {summary}.")
    return dataset
