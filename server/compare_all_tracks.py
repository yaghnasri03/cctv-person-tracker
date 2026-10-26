
import cv2
from pathlib import Path

VIDEO_PATH = Path("test_video.mp4")

# Frames around the suspected identity transition:
# Track 6 disappears around frame 496.
# Track 101 first appears around frame 536.
FRAMES_TO_CHECK = [488, 490, 492, 503, 510, 536, 540, 544]

OUTPUT_DIR = Path("transition_inspection")
OUTPUT_DIR.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {VIDEO_PATH.resolve()}"
    )

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {VIDEO_PATH.resolve()}")
print(f"Total frames: {total_frames}")

for frame_number in FRAMES_TO_CHECK:
    if frame_number >= total_frames:
        print(f"Skipping invalid frame {frame_number}")
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()

    if not success:
        print(f"Could not read frame {frame_number}")
        continue

    output_path = OUTPUT_DIR / f"frame_{frame_number:04d}.jpg"

    if cv2.imwrite(str(output_path), frame):
        print(f"Saved: {output_path}")

cap.release()
print("Done. No YOLO or InsightFace inference was run.")
