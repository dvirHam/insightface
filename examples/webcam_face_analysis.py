import argparse
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List

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
        if person_dir.name.startswith("_"):
            # Skip technical directories like _unlabeled.
            continue
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


def ensure_unlabeled_dir(known_dir: str) -> Path:
    unlabeled = Path(known_dir) / "_unlabeled"
    unlabeled.mkdir(parents=True, exist_ok=True)
    return unlabeled


def save_face_crop(
    frame: np.ndarray, bbox: np.ndarray, target_dir: Path, prefix: str = "face"
) -> Path | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = target_dir / f"{prefix}_{stamp}.jpg"
    cv2.imwrite(str(path), crop)
    return path


def estimate_yaw(face) -> float:
    pose = getattr(face, "pose", None)
    if pose is not None and len(pose) >= 2:
        return float(pose[1])
    kps = getattr(face, "kps", None)
    if kps is None or len(kps) < 3:
        return 0.0
    left_eye = kps[0].astype(np.float32)
    right_eye = kps[1].astype(np.float32)
    nose = kps[2].astype(np.float32)
    dl = float(np.linalg.norm(nose - left_eye))
    dr = float(np.linalg.norm(nose - right_eye))
    if dl + dr < 1e-6:
        return 0.0
    return ((dr - dl) / (dr + dl)) * 45.0


def image_sharpness(frame: np.ndarray, bbox: np.ndarray) -> float:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_quality_score(frame: np.ndarray, face, yaw: float) -> float:
    area = _face_area(face)
    sharp = image_sharpness(frame, face.bbox)
    frontal_bonus = max(0.0, 1.0 - abs(yaw) / 45.0)
    return (area / 10000.0) + (sharp / 200.0) + frontal_bonus


def find_cluster_index(
    embedding: np.ndarray, clusters: List[Dict[str, object]], threshold: float
) -> int | None:
    if not clusters:
        return None
    reps = np.stack([c["rep"] for c in clusters], axis=0)
    sims = reps @ embedding
    best_idx = int(np.argmax(sims))
    if float(sims[best_idx]) >= threshold:
        return best_idx
    return None


def should_save_better_shot(
    current_score: float,
    best_score: float,
    now_ts: float,
    last_save_ts: float,
    min_improve: float,
    min_interval_sec: float,
) -> bool:
    if current_score < (best_score + min_improve):
        return False
    if now_ts - last_save_ts < min_interval_sec:
        return False
    return True


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
    parser.add_argument(
        "--new-face-threshold",
        type=float,
        default=0.50,
        help="Cosine similarity threshold for deciding if a face is new/unseen.",
    )
    parser.add_argument(
        "--front-yaw-max",
        type=float,
        default=15.0,
        help="Absolute yaw <= this is considered front face.",
    )
    parser.add_argument(
        "--side-yaw-min",
        type=float,
        default=25.0,
        help="Absolute yaw >= this is considered side face.",
    )
    parser.add_argument(
        "--min-score-improve",
        type=float,
        default=0.25,
        help="Save only if quality score improves by at least this amount.",
    )
    parser.add_argument(
        "--min-save-interval-sec",
        type=float,
        default=2.5,
        help="Minimum seconds between saves for same identity/angle.",
    )
    args = parser.parse_args()

    available = find_available_cameras(args.max_cam_id, args.backend)
    print(f"Detected camera indexes: {available if available else 'none'}")

    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=args.ctx_id, det_size=(args.det_size, args.det_size))
    gallery_embeds, gallery_names = load_known_gallery(app, args.known_dir)
    unlabeled_dir = ensure_unlabeled_dir(args.known_dir)
    unknown_clusters: List[Dict[str, object]] = []
    print(f"Known identities loaded: {len(gallery_names)}")
    print(f"Unlabeled captures folder: {unlabeled_dir}")

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
    print("Controls: [n]=next camera, [p]=previous camera, [r]=reload known faces, [q]=quit")

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
            emb = face.normed_embedding.astype(np.float32)
            yaw = estimate_yaw(face)
            is_known = False
            if gallery_embeds.shape[0] > 0:
                sims = gallery_embeds @ emb
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                if best_sim >= args.match_threshold:
                    label = f"{gallery_names[best_idx]} ({best_sim:.2f})"
                    is_known = True
                else:
                    label = f"Unknown ({best_sim:.2f})"

            if not is_known:
                cluster_idx = find_cluster_index(
                    emb, unknown_clusters, args.new_face_threshold
                )
                if cluster_idx is None:
                    cluster_id = len(unknown_clusters)
                    saved_path = save_face_crop(
                        frame, face.bbox, unlabeled_dir, prefix=f"unknown_{cluster_id}_first"
                    )
                    unknown_clusters.append(
                        {
                            "rep": emb.copy(),
                            "front_score": -1.0,
                            "side_score": -1.0,
                            "front_saved_ts": 0.0,
                            "side_saved_ts": 0.0,
                            "front_path": None,
                            "side_path": None,
                        }
                    )
                    if saved_path is not None:
                        print(f"New face captured (first): {saved_path}")
                else:
                    cluster = unknown_clusters[cluster_idx]
                    # Slowly update representative embedding for stable clustering.
                    cluster["rep"] = (
                        0.9 * cluster["rep"] + 0.1 * emb
                    ) / (np.linalg.norm(0.9 * cluster["rep"] + 0.1 * emb) + 1e-12)
                    score = face_quality_score(frame, face, yaw)
                    abs_yaw = abs(yaw)
                    now_ts = time.time()
                    if abs_yaw <= args.front_yaw_max and should_save_better_shot(
                        score,
                        float(cluster["front_score"]),
                        now_ts,
                        float(cluster["front_saved_ts"]),
                        args.min_score_improve,
                        args.min_save_interval_sec,
                    ):
                        saved_path = save_face_crop(
                            frame,
                            face.bbox,
                            unlabeled_dir,
                            prefix=f"unknown_{cluster_idx}_front",
                        )
                        if saved_path is not None:
                            cluster["front_score"] = score
                            cluster["front_saved_ts"] = now_ts
                            cluster["front_path"] = str(saved_path)
                            print(f"Better front face saved: {saved_path}")
                    if abs_yaw >= args.side_yaw_min and should_save_better_shot(
                        score,
                        float(cluster["side_score"]),
                        now_ts,
                        float(cluster["side_saved_ts"]),
                        args.min_score_improve,
                        args.min_save_interval_sec,
                    ):
                        saved_path = save_face_crop(
                            frame,
                            face.bbox,
                            unlabeled_dir,
                            prefix=f"unknown_{cluster_idx}_side",
                        )
                        if saved_path is not None:
                            cluster["side_score"] = score
                            cluster["side_saved_ts"] = now_ts
                            cluster["side_path"] = str(saved_path)
                            print(f"Better side face saved: {saved_path}")
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                output,
                f"{label} | yaw:{yaw:.1f}",
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
        if key == ord("r"):
            gallery_embeds, gallery_names = load_known_gallery(app, args.known_dir)
            print(f"Reloaded known identities: {len(gallery_names)}")
            continue
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
