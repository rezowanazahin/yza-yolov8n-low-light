"""Convert the ExDark dataset to YOLO format.

Expected input layout (as downloaded from the official ExDark release):
    <exdark_root>/ExDark/<ClassName>/<image>.jpg|png ...
    <exdark_root>/ExDark_Anno/<ClassName>/<image>.jpg.txt ...
    <exdark_root>/imageclasslist.txt        (optional, gives the official train/val/test split)

Annotation lines look like:  Bicycle l t w h 0 0 0 0 0 0 0   (pixel units, left/top/width/height)

Usage:
    python convert_exdark.py --src /path/to/ExDark_root --dst datasets/ExDark
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"]
CLASS_ID = {c.lower(): i for i, c in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".jpe"}


def read_split_file(src):
    """Return {image name (lower): 'train'|'val'|'test'} from imageclasslist.txt, or {}."""
    path = src / "imageclasslist.txt"
    if not path.exists():
        return {}
    split_names = {1: "train", 2: "val", 3: "test"}
    out = {}
    for line in path.read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 5 and parts[4].isdigit():
            out[parts[0].lower()] = split_names.get(int(parts[4]), "train")
    return out


def find_annotation(anno_dir, image_path):
    for name in (image_path.name + ".txt", image_path.stem + ".txt"):
        p = anno_dir / name
        if p.exists():
            return p
    return None


def to_yolo_lines(anno_path, width, height):
    lines = []
    for raw in anno_path.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("%"):
            continue
        parts = raw.split()
        cls = CLASS_ID.get(parts[0].lower())
        if cls is None or len(parts) < 5:
            continue
        left, top, w, h = (float(v) for v in parts[1:5])
        xc, yc = (left + w / 2) / width, (top + h / 2) / height
        lines.append(f"{cls} {min(max(xc, 0), 1):.6f} {min(max(yc, 0), 1):.6f} "
                     f"{min(w / width, 1):.6f} {min(h / height, 1):.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder that contains ExDark/ and ExDark_Anno/")
    ap.add_argument("--dst", default="datasets/ExDark")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    splits = read_split_file(src)
    rng = random.Random(args.seed)
    counts = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    for cls_dir in sorted((src / "ExDark").iterdir()):
        if not cls_dir.is_dir():
            continue
        anno_dir = src / "ExDark_Anno" / cls_dir.name
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            anno = find_annotation(anno_dir, img)
            if anno is None:
                skipped += 1
                continue
            with Image.open(img) as im:
                w, h = im.size
            lines = to_yolo_lines(anno, w, h)
            if not lines:
                skipped += 1
                continue
            split = splits.get(img.name.lower())
            if split is None:  # no official list: random 70/15/15
                r = rng.random()
                split = "train" if r < 0.70 else "val" if r < 0.85 else "test"
            (dst / "images" / split).mkdir(parents=True, exist_ok=True)
            (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst / "images" / split / img.name)
            (dst / "labels" / split / (img.stem + ".txt")).write_text("\n".join(lines) + "\n")
            counts[split] += 1

    print("converted:", counts, "| skipped (no usable annotation):", skipped)


if __name__ == "__main__":
    main()
