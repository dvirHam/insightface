import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
from insightface.app import FaceAnalysis


BACKENDS = {
    "any": cv2.CAP_ANY,
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
}


def open_camera(index: int, width: int, height: int, backend: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return cap
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def open_camera_with_fallback(
    index: int, width: int, height: int, preferred_backend: str
) -> tuple[cv2.VideoCapture, str]:
    ordered = [preferred_backend] + [k for k in BACKENDS.keys() if k != preferred_backend]
    for backend_name in ordered:
        cap = open_camera(index, width, height, BACKENDS[backend_name])
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        if ok:
            return cap, backend_name
        cap.release()
    return cv2.VideoCapture(), "none"


def find_available_cameras(max_index: int = 10, preferred_backend: str = "any") -> List[int]:
    available = []
    for idx in range(max_index + 1):
        cap, _ = open_camera_with_fallback(idx, 640, 480, preferred_backend)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


def _face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox.astype(int)
    return max(0, x2 - x1) * max(0, y2 - y1)


def load_known_gallery(app: FaceAnalysis, known_dir: str) -> tuple[np.ndarray, List[str]]:
    root = Path(known_dir)
    if not root.exists():
        print(f"Known faces directory not found: {root}. Running without names.")
        return np.empty((0, 512), dtype=np.float32), []

    embeddings: List[np.ndarray] = []
    names: List[str] = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        person_embeds = []
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in image_exts:
                continue
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Skipping unreadable image: {image_path}")
                continue
            faces = app.get(img)
            if not faces:
                print(f"No face detected in: {image_path}")
                continue
            face = max(faces, key=_face_area)
            person_embeds.append(face.normed_embedding.astype(np.float32))

        if person_embeds:
            mean_embed = np.mean(np.stack(person_embeds, axis=0), axis=0)
            norm = np.linalg.norm(mean_embed) + 1e-12
            mean_embed = (mean_embed / norm).astype(np.float32)
            embeddings.append(mean_embed)
            names.append(person_dir.name)
            print(f"Loaded {len(person_embeds)} sample(s) for '{person_dir.name}'")
        else:
            print(f"No usable face images for '{person_dir.name}'")

    if not embeddings:
        print("No known identities loaded. Running without names.")
        return np.empty((0, 512), dtype=np.float32), []
    return np.stack(embeddings, axis=0), names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time InsightFace using laptop/external cameras."
    )
    parser.add_argument("--cam-id", type=int, default=0, help="Initial camera index.")
    parser.add_argument(
        "--max-cam-id",
        type=int,
        default=6,
        help="Maximum camera index checked for switching/listing.",
    )
    parser.add_argument(
        "--ctx-id",
        type=int,
        default=0,
        help="Use -1 for CPU, >=0 for GPU context.",
    )
    parser.add_argument(
        "--det-size", type=int, default=640, help="Face detector input resolution."
    )
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument(
        "--backend",
        type=str,
        default="any",
        choices=["any", "msmf", "dshow"],
        help="Preferred camera backend.",
    )
    parser.add_argument(
        "--known-dir",
        type=str,
        default="known_faces",
        help="Directory with known identities: known_faces/<name>/*.jpg",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.35,
        help="Cosine similarity threshold to assign a known name.",
    )
    args = parser.parse_args()

    available = find_available_cameras(args.max_cam_id, args.backend)
    print(f"Detected camera indexes: {available if available else 'none'}")

    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=args.ctx_id, det_size=(args.det_size, args.det_size))
    gallery_embeds, gallery_names = load_known_gallery(app, args.known_dir)
    print(f"Known identities loaded: {len(gallery_names)}")

    cam_id = args.cam_id
    cap, backend_name = open_camera_with_fallback(
        cam_id, args.width, args.height, args.backend
    )
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {cam_id}. "
            "Try --cam-id 1 (or 2), and check Windows Camera privacy settings."
        )
    print(f"Using camera index: {cam_id} (backend: {backend_name})")
    print("Controls: [n]=next camera, [p]=previous camera, [q]=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print(f"Failed to read frame from camera {cam_id}")
            break

        faces = app.get(frame)
        output = app.draw_on(frame, faces)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            label = "Unknown"
            if gallery_embeds.shape[0] > 0:
                emb = face.normed_embedding.astype(np.float32)
                sims = gallery_embeds @ emb
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                if best_sim >= args.match_threshold:
                    label = f"{gallery_names[best_idx]} ({best_sim:.2f})"
                else:
                    label = f"Unknown ({best_sim:.2f})"
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                output,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            output,
            f"Camera: {cam_id} | Faces: {len(faces)}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("InsightFace Webcam", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("n"), ord("p")):
            next_id = cam_id + 1 if key == ord("n") else cam_id - 1
            if next_id < 0:
                next_id = args.max_cam_id
            elif next_id > args.max_cam_id:
                next_id = 0

            new_cap, new_backend_name = open_camera_with_fallback(
                next_id, args.width, args.height, args.backend
            )
            if new_cap.isOpened():
                cap.release()
                cap = new_cap
                cam_id = next_id
                print(f"Switched to camera index: {cam_id} (backend: {new_backend_name})")
            else:
                print(f"Camera index {next_id} is not available.")
                new_cap.release()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
