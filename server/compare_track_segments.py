import cv2
import numpy as np
from PIL import Image

from ultralytics import YOLO
from person_matcher import PersonMatcher


VIDEO_PATH = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\test video 2.mp4"

TRACKS_TO_CHECK = {
    2,
    178,
    187
}


print("Loading YOLOv8...")

model = YOLO("yolov8n.pt")

print("Loading OSNet...")

matcher = PersonMatcher()

print("Starting tracking...")

results = model.track(
    source=VIDEO_PATH,
    tracker="bytetrack.yaml",
    persist=True,
    stream=True,
    verbose=False
)


# ---------------------------------------------------------
# Store one representative crop for each requested track
# ---------------------------------------------------------

track_crops = {}

track_frames = {}

for frame_number, result in enumerate(results):

    if frame_number > 1100:
        break

    if result.orig_img is None:
        continue

    if result.boxes is None:
        continue

    if result.boxes.id is None:
        continue

    frame = result.orig_img

    ids = (
        result.boxes.id
        .int()
        .cpu()
        .tolist()
    )

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
        .astype(int)
    )

    for i, track_id in enumerate(ids):

        if track_id not in TRACKS_TO_CHECK:
            continue

        if track_id in track_crops:
            continue

        x1, y1, x2, y2 = boxes[i]

        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue

        track_crops[track_id] = crop.copy()
        track_frames[track_id] = frame_number

        print(
            f"Collected Track {track_id} "
            f"at frame {frame_number}"
        )


# ---------------------------------------------------------
# Check what we collected
# ---------------------------------------------------------

print()
print("==============================")
print("COLLECTED TRACKS")
print("==============================")

for track_id in sorted(
    TRACKS_TO_CHECK
):

    if track_id in track_crops:

        crop = track_crops[
            track_id
        ]

        print(
            f"Track {track_id}: "
            f"frame={track_frames[track_id]}, "
            f"crop={crop.shape}"
        )

    else:

        print(
            f"Track {track_id}: NOT FOUND"
        )


# ---------------------------------------------------------
# Extract OSNet features
# ---------------------------------------------------------

features = {}

print()
print("==============================")
print("EXTRACTING OSNET FEATURES")
print("==============================")

for track_id in sorted(
    track_crops
):

    crop = track_crops[
        track_id
    ]

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    crop_pil = Image.fromarray(
        crop_rgb
    )

    feature = (
        matcher.extract_features(
            crop_pil
        )
    )

    features[track_id] = feature

    print(
        f"Track {track_id}: "
        f"feature extracted"
    )


# ---------------------------------------------------------
# Compare tracks
# ---------------------------------------------------------

print()
print("==============================")
print("TRACK SIMILARITIES")
print("==============================")


def compare(track_a, track_b):

    if (
        track_a not in features
        or
        track_b not in features
    ):

        print(
            f"Track {track_a} vs "
            f"Track {track_b}: "
            f"missing"
        )

        return

    similarity = (
        matcher.compare_features(
            features[track_a],
            features[track_b]
        )
    )

    print(
        f"Track {track_a} vs "
        f"Track {track_b}: "
        f"{similarity:.4f}"
    )


compare(2, 178)

compare(2, 187)

compare(178, 187)


print()
print("==============================")
print("DONE")
print("==============================")