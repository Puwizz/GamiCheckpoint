"""Build a leakage-resistant six-step dataset from named capture sessions.

The source layout is expected to be::

    1_New_Dataset/
      noon_session/Step1/*.jpg
      ...
      night_session/Step6/*.jpg

One complete session is copied to ``train`` and a different complete session
is copied to ``val``. This is intentionally different from random image
splitting: adjacent capture frames must not leak across the two splits.
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

# Add project root and src to python path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from step_config import STEP_CLASSES
from utils import safe_rmtree


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    root = ROOT
    parser = argparse.ArgumentParser(
        description="Prepare a six-step baseline with train/validation split by session."
    )
    parser.add_argument("--source-dir", type=Path, default=root / "data" / "sessions")
    parser.add_argument("--train-session", required=True)
    parser.add_argument("--val-session", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-images-per-step",
        type=int,
        default=30,
        help="Reject incomplete sessions before any files are copied.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory. Use only for generated experiment data.",
    )
    return parser.parse_args()


def image_files(directory):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def find_step_directory(session_dir, step_name):
    """Resolve ``Step1`` and legacy names such as ``Step1_noon`` safely."""
    exact = session_dir / step_name
    if exact.is_dir():
        return exact

    aliases = sorted(
        path for path in session_dir.glob(f"{step_name}_*") if path.is_dir()
    )
    if len(aliases) == 1:
        return aliases[0]
    if len(aliases) > 1:
        names = ", ".join(path.name for path in aliases)
        raise ValueError(
            f"Ambiguous directory names for {session_dir.name}/{step_name}: {names}"
        )
    return None


def collect_session(source_dir, session_name, min_images_per_step):
    session_dir = source_dir / session_name
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    by_class = {}
    for class_name in STEP_CLASSES:
        step_name = f"Step{class_name.removeprefix('Chk')}"
        step_dir = find_step_directory(session_dir, step_name)
        if step_dir is None:
            raise FileNotFoundError(
                f"Missing {step_name} for session {session_name}: {session_dir / step_name}"
            )
        files = image_files(step_dir)
        if len(files) < min_images_per_step:
            raise ValueError(
                f"{session_name}/{step_name} has {len(files)} images; "
                f"need at least {min_images_per_step}."
            )
        by_class[class_name] = files
    return by_class


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_exact_cross_split_duplicates(train_images, val_images):
    train_hashes = {sha256(path): path for paths in train_images.values() for path in paths}
    duplicates = [
        (train_hashes[sha256(path)], path)
        for paths in val_images.values()
        for path in paths
        if sha256(path) in train_hashes
    ]
    if duplicates:
        examples = ", ".join(f"{left.name} == {right.name}" for left, right in duplicates[:3])
        raise ValueError(
            "Exact duplicate files were found across sessions. "
            f"Remove duplicates before evaluation. Examples: {examples}"
        )


def copy_split(output_dir, split, session_name, images_by_class):
    rows = []
    for class_name, images in images_by_class.items():
        destination_dir = output_dir / split / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        for index, image_path in enumerate(images, start=1):
            # Prefixing prevents name collisions while retaining the original name.
            destination_name = f"{session_name}__{index:03d}__{image_path.name}"
            destination_path = destination_dir / destination_name
            shutil.copy2(image_path, destination_path)
            rows.append(
                {
                    "split": split,
                    "session": session_name,
                    "class_name": class_name,
                    "source_path": str(image_path.resolve()),
                    "output_path": str(destination_path.relative_to(output_dir)),
                    "sha256": sha256(image_path),
                }
            )
    return rows


def main():
    args = parse_args()
    if args.train_session == args.val_session:
        raise ValueError("--train-session and --val-session must be different.")
    if args.min_images_per_step < 1:
        raise ValueError("--min-images-per-step must be at least 1.")

    train_images = collect_session(
        args.source_dir, args.train_session, args.min_images_per_step
    )
    val_images = collect_session(
        args.source_dir, args.val_session, args.min_images_per_step
    )
    assert_no_exact_cross_split_duplicates(train_images, val_images)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {args.output_dir}. "
                "Use a new directory or pass --overwrite."
            )
        safe_rmtree(args.output_dir)

    rows = []
    rows.extend(copy_split(args.output_dir, "train", args.train_session, train_images))
    rows.extend(copy_split(args.output_dir, "val", args.val_session, val_images))

    manifest_path = args.output_dir / "session_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["split", "session", "class_name", "source_path", "output_path", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_dir": str(args.source_dir.resolve()),
        "train_session": args.train_session,
        "val_session": args.val_session,
        "classes": list(STEP_CLASSES),
        "counts": {
            "train": {class_name: len(paths) for class_name, paths in train_images.items()},
            "val": {class_name: len(paths) for class_name, paths in val_images.items()},
        },
    }
    summary_path = args.output_dir / "session_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Prepared session baseline dataset")
    print(f"  train={args.train_session} ({sum(len(paths) for paths in train_images.values())} images)")
    print(f"  val={args.val_session} ({sum(len(paths) for paths in val_images.values())} images)")
    print(f"  manifest={manifest_path}")
    print(f"  summary={summary_path}")


if __name__ == "__main__":
    main()
