import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


VIDEO_PATH = Path(
    r"uploads\2328a724-013a-489a-8d73-80d7a773d2e0\test video 2.mp4"
)

OUTPUT_DIR = Path("tracking_transition_diagnostic")

MODEL_PATH = "yolov8n.pt"
TRACKER_CONFIG = "bytetrack.yaml"

# Process from frame 0 to approximately 20 seconds.
END_SECONDS = 20.5

# Save images at these timestamps.
SAVE_TIMES = [16, 17, 18, 19, 20]


def main():
    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    frames_dir = OUTPUT_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)

    csv_path = OUTPUT_DIR / "tracking_results.csv"

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Opening video...")
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Could not open the video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    end_frame = min(
        int(END_SECONDS * fps),
        total_frames
    )

    print(f"FPS: {fps:.3f}")
    print(f"Total video frames: {total_frames}")
    print(f"Processing frames 0 to {end_frame - 1}")

    cap.release()

    results = model.track(
        source=str(VIDEO_PATH),
        tracker=TRACKER_CONFIG,
        persist=True,
        stream=True,
        verbose=False,
        classes=[0],
        conf=0.25,
        vid_stride=1
    )

    save_frames = {
        round(seconds * fps): seconds
        for seconds in SAVE_TIMES
    }

    rows = []
    processed_frames = 0

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame",
                "seconds",
                "track_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2"
            ]
        )

        writer.writeheader()

        for frame_number, result in enumerate(results):
            if frame_number >= end_frame:
                break

            frame = result.orig_img

            if frame is None:
                continue

            processed_frames += 1

            boxes = result.boxes
            annotated = frame.copy()

            if boxes is not None and len(boxes) > 0:
                ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(boxes)
                )

                coordinates = (
                    boxes.xyxy.cpu().numpy()
                )

                confidences = (
                    boxes.conf.cpu().tolist()
                )

                for track_id, coords, confidence in zip(
                    ids,
                    coordinates,
                    confidences
                ):
                    if track_id is None:
                        continue

                    x1, y1, x2, y2 = map(int, coords)

                    seconds = frame_number / fps

                    if seconds >= 15.5:
                        row = {
                            "frame": frame_number,
                            "seconds": round(seconds, 3),
                            "track_id": track_id,
                            "confidence": round(confidence, 4),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }

                        writer.writerow(row)
                        rows.append(row)

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        annotated,
                        f"ID {track_id}",
                        (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2
                    )

            if frame_number in save_frames:
                seconds = save_frames[frame_number]

                image_path = (
                    frames_dir
                    / f"tracking_{seconds:02d}s.jpg"
                )

                cv2.imwrite(
                    str(image_path),
                    annotated
                )

                print(f"Saved {image_path}")

            if processed_frames % 150 == 0:
                print(
                    f"Processed {processed_frames}/"
                    f"{end_frame} frames"
                )

    print("\nDiagnostic completed.")
    print(f"Processed frames: {processed_frames}")
    print(f"CSV: {csv_path}")
    print(f"Images: {frames_dir}")

    print("\nTrack IDs observed around 16-20 seconds:")

    unique_ids = sorted({
        row["track_id"]
        for row in rows
    })

    print(unique_ids)


if __name__ == "__main__":
    main()