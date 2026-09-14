from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


class _StreamingJSON:
    """Read top-level JSON arrays without loading the source file at once."""

    _CHUNK_SIZE = 1024 * 1024

    def __init__(self, file):
        self.file = file
        self.buffer = ""
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _fill(self):
        if self.eof:
            return
        chunk = self.file.read(self._CHUNK_SIZE)
        if chunk:
            self.buffer += chunk
        else:
            self.eof = True

    def _skip_whitespace(self):
        while True:
            stripped = self.buffer.lstrip()
            if stripped:
                self.buffer = stripped
                return
            if self.eof:
                return
            self._fill()

    def _seek_array(self, key: str):
        marker = f'"{key}"'
        while True:
            marker_start = self.buffer.find(marker)
            if marker_start >= 0:
                self.buffer = self.buffer[marker_start + len(marker) :]
                self._skip_whitespace()
                if not self.buffer.startswith(":"):
                    raise ValueError(f"Invalid LVIS JSON field: {key}")
                self.buffer = self.buffer[1:]
                self._skip_whitespace()
                if not self.buffer.startswith("["):
                    raise ValueError(f"LVIS JSON field is not an array: {key}")
                self.buffer = self.buffer[1:]
                return

            if self.eof:
                raise KeyError(f"LVIS JSON field not found: {key}")
            self.buffer = self.buffer[-(len(marker) - 1) :]
            self._fill()

    def _next_value(self):
        while True:
            self._skip_whitespace()
            try:
                value, end = self.decoder.raw_decode(self.buffer)
                self.buffer = self.buffer[end:]
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self._fill()

    def iter_array(self, key: str):
        self._seek_array(key)
        while True:
            self._skip_whitespace()
            if self.buffer.startswith("]"):
                self.buffer = self.buffer[1:]
                return

            yield self._next_value()
            self._skip_whitespace()
            if self.buffer.startswith(","):
                self.buffer = self.buffer[1:]
                continue
            if self.buffer.startswith("]"):
                self.buffer = self.buffer[1:]
                return
            raise ValueError(f"Invalid LVIS JSON array: {key}")


def _iter_lvis_array(json_path: Path, key: str):
    with json_path.open("r", encoding="utf-8") as file:
        parser = _StreamingJSON(file)
        yield from parser.iter_array(key)


def _file_name_from_lvis_image(image_info: dict[str, Any]) -> str:
    file_name = image_info.get("file_name")
    if file_name:
        return str(file_name).replace("\\", "/").rsplit("/", 1)[-1]

    coco_url = image_info.get("coco_url")
    if coco_url:
        url_path = urlparse(str(coco_url)).path.replace("\\", "/")
        file_name = PurePosixPath(url_path).name
        if file_name:
            return file_name

    raise KeyError(
        "LVIS image record must contain either 'file_name' or a usable "
        f"'coco_url'. Image id: {image_info.get('id')}"
    )


def convert_lvis_json(
    input_json: str,
    output_json: str,
    image_root: str | None = None,
    keep_missing: bool = False,
) -> None:
    """Convert an LVIS annotation file to the COCO format used by SAM3."""
    input_path = Path(input_json)
    output_path = Path(output_json)
    image_root_path = Path(image_root) if image_root else input_path.parent

    print(f"Converting LVIS JSON: {input_path}", flush=True)
    print(f"Image root: {image_root_path}", flush=True)

    images = []
    image_ids = set()
    missing_images = 0
    for index, image_info in enumerate(_iter_lvis_array(input_path, "images"), 1):
        normalized_image = dict(image_info)
        normalized_image["file_name"] = _file_name_from_lvis_image(normalized_image)
        exists = (image_root_path / normalized_image["file_name"]).is_file()
        if not keep_missing and not exists:
            missing_images += 1
            continue
        images.append(normalized_image)
        image_ids.add(normalized_image["id"])
        if index % 10000 == 0:
            print(f"  processed image records: {index}", flush=True)

    if not images:
        raise FileNotFoundError(
            f"No usable images found in {input_path} under {image_root_path}."
        )

    annotations = []
    for index, annotation in enumerate(
        _iter_lvis_array(input_path, "annotations"), 1
    ):
        if annotation["image_id"] not in image_ids:
            continue
        normalized_annotation = dict(annotation)
        normalized_annotation.setdefault("iscrowd", 0)
        annotations.append(normalized_annotation)
        if index % 100000 == 0:
            print(f"  processed annotation records: {index}", flush=True)

    categories = list(_iter_lvis_array(input_path, "categories"))
    converted = {
        "info": {
            "description": "LVIS converted to COCO-compatible JSON for SAM3 KD",
            "source": str(input_path),
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "licenses": [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_output.open("w", encoding="utf-8") as file:
        json.dump(converted, file, ensure_ascii=False)
    temp_output.replace(output_path)

    print(
        f"Done: images={len(images)}, annotations={len(annotations)}, "
        f"categories={len(categories)}, skipped_missing_images={missing_images}",
        flush=True,
    )
    print(f"Output: {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LVIS annotations to COCO-compatible JSON."
    )
    parser.add_argument("--input", required=True, help="Source LVIS JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--image-root",
        default=None,
        help="Directory containing local images; defaults to input JSON directory",
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep image records whose local files are missing",
    )
    args = parser.parse_args()
    convert_lvis_json(
        input_json=args.input,
        output_json=args.output,
        image_root=args.image_root,
        keep_missing=args.keep_missing,
    )


if __name__ == "__main__":
    main()
