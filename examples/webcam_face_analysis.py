import argparse
from typing import List

import cv2
from insightface.app import FaceAnalysis


def find_available_cameras(max_index: int = 10) -> List[int]:
    available = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(idx)
        cap.release()
    return available


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return cap
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


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
    args = parser.parse_args()

    available = find_available_cameras(args.max_cam_id)
    print(f"Detected camera indexes: {available if available else 'none'}")

    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=args.ctx_id, det_size=(args.det_size, args.det_size))

    cam_id = args.cam_id
    cap = open_camera(cam_id, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_id}")
    print(f"Using camera index: {cam_id}")
    print("Controls: [n]=next camera, [p]=previous camera, [q]=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print(f"Failed to read frame from camera {cam_id}")
            break

        faces = app.get(frame)
        output = app.draw_on(frame, faces)
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

            new_cap = open_camera(next_id, args.width, args.height)
            if new_cap.isOpened():
                cap.release()
                cap = new_cap
                cam_id = next_id
                print(f"Switched to camera index: {cam_id}")
            else:
                print(f"Camera index {next_id} is not available.")
                new_cap.release()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
