from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image


IMAGE_DIR = Path(r"F:\data\OpenDataLab___LVIS_v1_dot_0\raw\LVIS_v1.0\train2017")
ANNOTATION_FILE = Path(
    r"F:\data\OpenDataLab___LVIS_v1_dot_0\raw\LVIS_v1.0\annotations\lvis_v1_train.json"
)
DEFAULT_OUTPUT_FILE = Path(__file__).with_name("lvis_random_ground_truth.jpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize LVIS ground-truth bounding boxes on one random image."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for selecting a reproducible random image.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def get_image_filename(image_record: dict[str, Any]) -> str:
    """Resolve LVIS image records with either file_name or COCO URL metadata."""
    file_name = image_record.get("file_name")
    if file_name:
        return Path(file_name).name

    coco_url = image_record.get("coco_url")
    if coco_url:
        return Path(urlparse(coco_url).path).name

    return f"{int(image_record['id']):012d}.jpg"


def is_valid_bbox(annotation: dict[str, Any]) -> bool:
    bbox = annotation.get("bbox", ())
    return len(bbox) == 4 and float(bbox[2]) > 0 and float(bbox[3]) > 0


def select_random_annotated_image(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_ids_with_boxes = {
        int(annotation["image_id"])
        for annotation in annotations
        if is_valid_bbox(annotation)
    }
    candidates = [
        image for image in images if int(image["id"]) in image_ids_with_boxes
    ]
    if not candidates:
        raise RuntimeError("No LVIS image with a valid bounding box was found.")

    # Some downloaded LVIS subsets can omit individual COCO image files.
    for image_record in rng.sample(candidates, k=len(candidates)):
        image_path = IMAGE_DIR / get_image_filename(image_record)
        if not image_path.is_file():
            continue
        image_id = int(image_record["id"])
        image_annotations = [
            annotation
            for annotation in annotations
            if int(annotation["image_id"]) == image_id and is_valid_bbox(annotation)
        ]
        if image_annotations:
            return image_record, image_annotations

    raise FileNotFoundError(
        f"No annotated LVIS image could be found under {IMAGE_DIR}."
    )


def draw_ground_truth_boxes(
    image_path: Path,
    image_record: dict[str, Any],
    annotations: list[dict[str, Any]],
    category_names: dict[int, str],
) -> np.ndarray:
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    image_height, image_width = image_rgb.shape[:2]
    expected_size = (int(image_record["width"]), int(image_record["height"]))
    actual_size = (image_width, image_height)
    if actual_size != expected_size:
        raise ValueError(
            f"Image size {actual_size} does not match LVIS metadata {expected_size}."
        )

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for annotation in annotations:
        x, y, width, height = (float(value) for value in annotation["bbox"])
        x1 = max(0, min(image_width - 1, round(x)))
        y1 = max(0, min(image_height - 1, round(y)))
        x2 = max(0, min(image_width - 1, round(x + width)))
        y2 = max(0, min(image_height - 1, round(y + height)))
        if x2 <= x1 or y2 <= y1:
            continue

        category_id = int(annotation["category_id"])
        category_name = category_names.get(category_id, str(category_id))
        label = f"{category_id}: {category_name}"
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_bgr,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return image_bgr


def main() -> None:
    args = parse_args()
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {IMAGE_DIR}")
    if not ANNOTATION_FILE.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {ANNOTATION_FILE}")

    with ANNOTATION_FILE.open("r", encoding="utf-8") as file:
        lvis_data = json.load(file)

    category_names = {
        int(category["id"]): str(category["name"])
        for category in lvis_data["categories"]
    }
    image_record, annotations = select_random_annotated_image(
        lvis_data["images"],
        lvis_data["annotations"],
        random.Random(args.seed),
    )
    image_path = IMAGE_DIR / get_image_filename(image_record)
    visualization = draw_ground_truth_boxes(
        image_path,
        image_record,
        annotations,
        category_names,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), visualization):
        raise IOError(f"Failed to write visualization: {args.output}")

    print(f"image: {image_path}")
    print(f"image_id: {image_record['id']}")
    print(f"image_size: {image_record['width']}x{image_record['height']}")
    print(f"ground_truth_boxes: {len(annotations)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
