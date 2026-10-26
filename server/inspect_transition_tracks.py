
import cv2
from pathlib import Path
from ultralytics import YOLO

VIDEO_PATH = Path("test_video.mp4")
OUTPUT_DIR = Path("transition_inspection")
TARGET_FRAMES = {488, 490, 492, 503, 510, 536, 540, 544}

print("Loading YOLOv8n...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise FileNotFoundError(VIDEO_PATH.resolve())

# ByteTrack IDs depend on tracking history. Replay frames sequentially
# through the last selected frame, but run detection only on selected frames.
# This is much shorter than processing the full 1302-frame video.
last_frame = max(TARGET_FRAMES)

for frame_number in range(last_frame + 1):
    ok, frame = cap.read()
    if not ok:
        break

    if frame_number not in TARGET_FRAMES:
        continue

    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.25,
        verbose=False
    )[0]

    annotated = frame.copy()
    print(f"\nFrame {frame_number}:")

    if result.boxes is not None and result.boxes.id is not None:
        for box, track_id, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.id.cpu().numpy(),
            result.boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)

            print(
                f"  Track {track_id}: "
                f"bbox={[x1, y1, x2, y2]}, confidence={conf:.3f}"
            )

            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), (0, 255, 0), 2
            )
            cv2.putText(
                annotated, f"ID {track_id}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )

    output_path = OUTPUT_DIR / f"tracked_{frame_number:04d}.jpg"
    cv2.imwrite(str(output_path), annotated)

cap.release()
print("\nDone. Annotated frames saved in:", OUTPUT_DIR.resolve())
