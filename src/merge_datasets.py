"""Merge several YOLOv8-format helmet datasets into one with unified classes.

    python -m src.merge_datasets --inputs datasets/helmet datasets/helmet2 --out datasets/helmet_merged

Unified classes: 0 = helmet, 1 = no_helmet
"""
import argparse
import shutil
from pathlib import Path

import yaml

from src.utils import norm_name

HELMET = {"helmet", "with_helmet", "pakai_helm", "wearing_helmet"}
NO_HELMET = {"no_helmet", "nohelmet", "without_helmet", "tanpa_helm"}
UNIFIED = ["helmet", "no_helmet"]
SPLITS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_map(names):
    """Return {old_class_id: new_class_id}; unknown classes are dropped."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    mapping = {}
    for idx, name in items:
        n = norm_name(str(name))
        if n in HELMET:
            mapping[int(idx)] = 0
        elif n in NO_HELMET:
            mapping[int(idx)] = 1
        else:
            print(f"  ! class '{name}' not recognised -> dropped")
    return mapping


def merge(inputs, out):
    out = Path(out)
    for s in ("train", "valid", "test"):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "valid": 0, "test": 0}
    for n, d in enumerate(inputs):
        d = Path(d)
        cfg = yaml.safe_load(open(d / "data.yaml"))
        mapping = build_map(cfg["names"])
        print(f"{d}: classes {cfg['names']} -> mapping {mapping}")
        if not mapping:
            raise SystemExit(f"No usable classes in {d}. Check its data.yaml.")
        for src_split, dst_split in SPLITS.items():
            img_dir = d / src_split / "images"
            if not img_dir.exists():
                continue
            for img in img_dir.iterdir():
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lbl = d / src_split / "labels" / (img.stem + ".txt")
                lines = []
                if lbl.exists():
                    for line in lbl.read_text().splitlines():
                        parts = line.split()
                        if len(parts) >= 5 and int(float(parts[0])) in mapping:
                            parts[0] = str(mapping[int(float(parts[0]))])
                            lines.append(" ".join(parts))
                stem = f"d{n}_{img.stem}"
                shutil.copy(img, out / dst_split / "images" / (stem + img.suffix))
                (out / dst_split / "labels" / (stem + ".txt")).write_text("\n".join(lines))
                counts[dst_split] += 1

    data = {"path": str(out.resolve()), "train": "train/images", "val": "valid/images",
            "names": {i: n for i, n in enumerate(UNIFIED)}}
    if counts["test"]:
        data["test"] = "test/images"
    yaml.safe_dump(data, open(out / "data.yaml", "w"), sort_keys=False)
    print(f"\nMerged images: {counts}\nWrote {out / 'data.yaml'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="datasets/helmet_merged")
    args = ap.parse_args()
    merge(args.inputs, args.out)


if __name__ == "__main__":
    main()