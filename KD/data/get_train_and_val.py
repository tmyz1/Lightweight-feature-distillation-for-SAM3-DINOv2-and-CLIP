import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pycocotools import mask as maskUtils


DEFAULT_SOURCE_ROOT = r"F:\data\SA-1B"
DEFAULT_SAMPLES_PER_FOLDER = 10000
DEFAULT_OUTPUT_NAME = "SA-train"
DEFAULT_JSON_NAME = "train.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OBJECT_CATEGORY = {"supercategory": "object", "id": 1, "name": "object"}

"""
搜寻所有的SA文件夹
"""
def find_sa_folders(source_root: Path, output_dir: Path) -> List[Path]:
    folders = []
    for path in sorted(source_root.iterdir()):
        if not path.is_dir():
            continue
        if path.resolve() == output_dir.resolve():
            continue
        if not path.name.startswith("sa_"):
            continue
        folders.append(path)
    return folders

"""
将文件夹下的image和json两两匹配
"""
def collect_image_json_pairs(folder: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for image_path in sorted(folder.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        json_path = image_path.with_suffix(".json")
        if json_path.exists():
            pairs.append((image_path, json_path))
    return pairs


def load_sa_json(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_output_image_name(folder: Path, image_path: Path) -> str:
    return f"{folder.name}_{image_path.name}"


def to_jsonable_rle(rle: Dict) -> Dict:
    rle = dict(rle)
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle

"""
mask解码
"""
def decode_and_reencode_mask(segmentation: Dict) -> Tuple[Dict, float, List[float]]:
    """Decode SA-1B RLE once, then re-encode as COCO-compatible compressed RLE."""
    mask = maskUtils.decode(segmentation)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    encoded = maskUtils.encode(np.asfortranarray(mask))
    encoded = to_jsonable_rle(encoded)
    area = float(maskUtils.area(encoded))
    bbox = [float(x) for x in maskUtils.toBbox(encoded).tolist()]
    return encoded, area, bbox


def build_coco_image(image_id: int, image_info: Dict, file_name: str) -> Dict:
    return {
        "license": image_info.get("license", 0),
        "file_name": file_name,
        "coco_url": image_info.get("coco_url", ""),
        "height": image_info.get("height"),
        "width": image_info.get("width"),
        "date_captured": image_info.get("date_captured", ""),
        "flickr_url": image_info.get("flickr_url", ""),
        "id": image_id,
    }


def build_coco_annotation(
    annotation_id: int,
    image_id: int,
    sa_ann: Dict,
    decode_masks: bool,
) -> Dict:
    segmentation = sa_ann["segmentation"]
    if decode_masks:
        segmentation, area, bbox = decode_and_reencode_mask(segmentation)
    else:
        segmentation = to_jsonable_rle(segmentation)
        area = float(sa_ann.get("area", maskUtils.area(segmentation)))
        bbox = [float(x) for x in sa_ann.get("bbox", maskUtils.toBbox(segmentation).tolist())]

    coco_ann = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": int(sa_ann.get("iscrowd", 0)),
        "image_id": image_id,
        "bbox": bbox,
        "category_id": OBJECT_CATEGORY["id"],
        "id": annotation_id,
    }

    for key in ("predicted_iou", "stability_score", "crop_box", "point_coords"):
        if key in sa_ann:
            coco_ann[key] = sa_ann[key]
    if "id" in sa_ann:
        coco_ann["sa1b_annotation_id"] = sa_ann["id"]

    return coco_ann


def build_train_dataset(
    source_root: Path,
    samples_per_folder: int,
    output_dir: Path,
    json_name: str,
    seed: int,
    overwrite: bool,
    decode_masks: bool,
) -> Dict:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists: {output_dir}. "
                "Use --overwrite if you want to rebuild it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    folders = find_sa_folders(source_root, output_dir)
    images = []
    annotations = []
    folder_stats = []
    image_id = 1
    annotation_id = 1

    for folder in folders:
        pairs = collect_image_json_pairs(folder)
        if len(pairs) == 0:
            folder_stats.append(
                {"folder": folder.name, "available": 0, "selected": 0, "warning": "no image/json pairs"}
            )
            continue

        selected_count = min(samples_per_folder, len(pairs))
        selected_pairs = rng.sample(pairs, selected_count)
        selected_pairs.sort(key=lambda item: item[0].name)

        for image_path, json_path in selected_pairs:
            out_image_name = make_output_image_name(folder, image_path)
            out_image_path = output_dir / out_image_name
            shutil.copy2(image_path, out_image_path)

            sa_json = load_sa_json(json_path)
            image_info = sa_json.get("image", {})
            coco_image = build_coco_image(image_id, image_info, out_image_name)
            coco_image["source_folder"] = folder.name
            coco_image["source_file_name"] = image_path.name
            coco_image["source_json_name"] = json_path.name
            if "image_id" in image_info:
                coco_image["sa1b_image_id"] = image_info["image_id"]
            images.append(coco_image)

            for sa_ann in sa_json.get("annotations", []):
                if "segmentation" not in sa_ann:
                    continue
                annotations.append(
                    build_coco_annotation(
                        annotation_id=annotation_id,
                        image_id=image_id,
                        sa_ann=sa_ann,
                        decode_masks=decode_masks,
                    )
                )
                annotation_id += 1

            print(f'image:{image_id} named {out_image_name} is finished')
            image_id += 1

        folder_stats.append(
            {"folder": folder.name, "available": len(pairs), "selected": selected_count}
        )

    train_json = {
        "info": {
            "description": "SA-1B sampled train split converted to COCO-style instance annotations",
            "source_root": str(source_root),
            "samples_per_folder": samples_per_folder,
            "num_source_folders": len(folders),
            "num_images": len(images),
            "num_annotations": len(annotations),
            "decode_masks": decode_masks,
            "folder_stats": folder_stats,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [OBJECT_CATEGORY],
    }

    json_path = output_dir / json_name
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(train_json, f, ensure_ascii=False)

    return train_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample SA-1B image/json pairs from each subfolder and build one COCO-style SA-train folder."
    )
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT, help="SA-1B upper folder, e.g. F:\\data\\test")
    parser.add_argument("--samples-per-folder", type=int, default=DEFAULT_SAMPLES_PER_FOLDER)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json-name", default=DEFAULT_JSON_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Delete old SA-train and rebuild it.")
    parser.add_argument(
        "--no-decode-masks",
        action="store_true",
        help="Keep SA-1B RLE directly. By default masks are decoded and re-encoded as COCO RLE.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"source root is not a folder: {source_root}")

    output_dir = source_root / args.output_name
    result = build_train_dataset(
        source_root=source_root,
        samples_per_folder=args.samples_per_folder,
        output_dir=output_dir,
        json_name=args.json_name,
        seed=args.seed,
        overwrite=args.overwrite,
        decode_masks=not args.no_decode_masks,
    )

    print(f"source_root: {source_root}")
    print(f"output_dir: {output_dir}")
    print(f"json_file: {output_dir / args.json_name}")
    print(f"num_source_folders: {result['info']['num_source_folders']}")
    print(f"num_images: {result['info']['num_images']}")
    print(f"num_annotations: {result['info']['num_annotations']}")
    print("folder_stats:")
    for stat in result["info"]["folder_stats"]:
        print(f"  {stat['folder']}: selected {stat['selected']} / available {stat['available']}")


if __name__ == "__main__":
    main()


