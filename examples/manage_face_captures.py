import argparse
import shutil
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_unlabeled_images(unlabeled_dir: Path) -> list[Path]:
    return sorted(
        [p for p in unlabeled_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


def draw_ui(img, file_name: str, idx: int, total: int):
    display = img.copy()
    lines = [
        f"[{idx + 1}/{total}] {file_name}",
        "Keys: [a]=assign name  [s]=skip  [d]=delete  [q]=quit",
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

        show = draw_ui(img, image_path.name, idx, len(images))
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
        if key == ord("a"):
            assign_image(image_path, known_dir)
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
