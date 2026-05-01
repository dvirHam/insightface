import argparse
import re
import shutil
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNKNOWN_GROUP_RE = re.compile(r"^unknown_(\d+)_")


def list_unlabeled_images(unlabeled_dir: Path) -> list[Path]:
    return sorted(
        [p for p in unlabeled_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


def image_group_key(image_path: Path) -> str:
    match = UNKNOWN_GROUP_RE.match(image_path.stem)
    if match:
        return f"unknown_{match.group(1)}"
    return image_path.stem


def files_in_group(image_path: Path, all_images: list[Path]) -> list[Path]:
    key = image_group_key(image_path)
    return [p for p in all_images if image_group_key(p) == key]


def draw_ui(img, file_name: str, idx: int, total: int, group_key: str, group_size: int):
    display = img.copy()
    lines = [
        f"[{idx + 1}/{total}] {file_name}",
        f"Group: {group_key} ({group_size} image(s))",
        "Keys: [a]=assign one  [g]=assign group  [s]=skip  [d]=delete one  [x]=delete group  [q]=quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            display,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30
    return display


def assign_image(image_path: Path, known_dir: Path):
    name = input(f"Enter person name for '{image_path.name}' (blank to cancel): ").strip()
    if not name:
        print("Assignment canceled.")
        return
    target_dir = known_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / image_path.name
    if target_path.exists():
        stem, suffix = image_path.stem, image_path.suffix
        i = 1
        while True:
            candidate = target_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                target_path = candidate
                break
            i += 1
    shutil.move(str(image_path), str(target_path))
    print(f"Assigned: {image_path.name} -> {target_path}")


def assign_group(images: list[Path], known_dir: Path):
    if not images:
        return
    name = input(
        f"Enter person name for group '{image_group_key(images[0])}' ({len(images)} images), blank to cancel: "
    ).strip()
    if not name:
        print("Group assignment canceled.")
        return
    target_dir = known_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for image_path in images:
        target_path = target_dir / image_path.name
        if target_path.exists():
            stem, suffix = image_path.stem, image_path.suffix
            i = 1
            while True:
                candidate = target_dir / f"{stem}_{i}{suffix}"
                if not candidate.exists():
                    target_path = candidate
                    break
                i += 1
        shutil.move(str(image_path), str(target_path))
        moved += 1
    print(f"Assigned group '{image_group_key(images[0])}' -> '{name}' ({moved} images)")


def delete_group(images: list[Path]):
    if not images:
        return
    group = image_group_key(images[0])
    for image_path in images:
        image_path.unlink(missing_ok=True)
    print(f"Deleted group '{group}' ({len(images)} images)")


def main():
    parser = argparse.ArgumentParser(
        description="Review and assign unlabeled face captures to known identities."
    )
    parser.add_argument(
        "--known-dir",
        type=str,
        default="known_faces",
        help="Root identity directory used by webcam_face_analysis.py",
    )
    args = parser.parse_args()

    known_dir = Path(args.known_dir)
    unlabeled_dir = known_dir / "_unlabeled"
    unlabeled_dir.mkdir(parents=True, exist_ok=True)

    print(f"Known folder: {known_dir}")
    print(f"Unlabeled folder: {unlabeled_dir}")

    idx = 0
    while True:
        images = list_unlabeled_images(unlabeled_dir)
        if not images:
            print("No unlabeled images left.")
            break
        if idx >= len(images):
            idx = 0
        image_path = images[idx]
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Unreadable image, deleting: {image_path}")
            image_path.unlink(missing_ok=True)
            continue

        group_images = files_in_group(image_path, images)
        group_key = image_group_key(image_path)
        show = draw_ui(
            img, image_path.name, idx, len(images), group_key=group_key, group_size=len(group_images)
        )
        cv2.imshow("Manage Face Captures", show)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            idx += 1
            continue
        if key == ord("d"):
            image_path.unlink(missing_ok=True)
            print(f"Deleted: {image_path.name}")
            continue
        if key == ord("x"):
            delete_group(group_images)
            continue
        if key == ord("a"):
            assign_image(image_path, known_dir)
            continue
        if key == ord("g"):
            assign_group(group_images, known_dir)
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
