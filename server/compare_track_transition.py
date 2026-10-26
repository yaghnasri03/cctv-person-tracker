
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from person_matcher import PersonMatcher

VIDEO_PATH = "test_video.mp4"
OUTPUT_PATH = "transition_features.npz"

SOURCE_TRACK = 6
SOURCE_FRAMES = {488, 489, 490, 491, 492}

CANDIDATE_FRAMES = {
    94: set(range(503, 514)),
    101: set(range(536, 546)),
}

LAST_FRAME = 545


def pack_features(feature_dict):
    """Convert {frame: feature} into aligned frame and feature arrays."""
    frames = sorted(feature_dict.keys())

    if not frames:
        return (
            np.array([], dtype=np.int32),
            np.empty((0, 512), dtype=np.float32),
        )

    features = np.stack([
        np.asarray(feature_dict[frame], dtype=np.float32).reshape(-1)
        for frame in frames
    ])

    return np.asarray(frames, dtype=np.int32), features


def compare(source, candidate, matcher):
    similarities = []

    for source_frame, source_feature in source.items():
        for candidate_frame, candidate_feature in candidate.items():
            similarity = matcher.compare_features(
                source_feature,
                candidate_feature,
            )

            similarities.append(
                (source_frame, candidate_frame, similarity)
            )

    return similarities


def print_summary(title, results):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not results:
        print("No feature pairs available for comparison.")
        return

    for source_frame, candidate_frame, similarity in results:
        print(
            f"{source_frame} -> {candidate_frame}: "
            f"{similarity:.4f}"
        )

    values = np.asarray(
        [result[2] for result in results],
        dtype=np.float32,
    )

    print(f"\nAverage : {np.mean(values):.4f}")
    print(f"Median  : {np.median(values):.4f}")
    print(f"Maximum : {np.max(values):.4f}")
    print(f"Minimum : {np.min(values):.4f}")


print("Loading YOLOv8...")
model = YOLO("yolov8n.pt")

print("Loading OSNet...")
matcher = PersonMatcher()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open video: {VIDEO_PATH}")
    raise SystemExit(1)

source_features = {}
candidate_features = {
    94: {},
    101: {},
}

frame_number = 0

print()
print("=" * 70)
print("RUNNING CONTINUOUS BYTE TRACK")
print("=" * 70)

while frame_number <= LAST_FRAME:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        verbose=False,
    )

    boxes = results[0].boxes if results else None

    if boxes is not None and boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()
        coordinates = boxes.xyxy.int().cpu().tolist()

        for track_id, box in zip(ids, coordinates):
            should_extract = (
                track_id == SOURCE_TRACK
                and frame_number in SOURCE_FRAMES
            ) or (
                track_id in CANDIDATE_FRAMES
                and frame_number in CANDIDATE_FRAMES[track_id]
            )

            if not should_extract:
                continue

            x1, y1, x2, y2 = box
            height, width = frame.shape[:2]

            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            image = Image.fromarray(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            )

            feature = np.asarray(
                matcher.extract_features(image),
                dtype=np.float32,
            ).reshape(-1)

            if track_id == SOURCE_TRACK:
                source_features[frame_number] = feature
            else:
                candidate_features[track_id][frame_number] = feature

            print(f"Captured Track {track_id}, frame {frame_number}")

    frame_number += 1

cap.release()

print()
print("=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print(f"Frames processed: {frame_number}")
print(f"Track 6 features: {len(source_features)}")
print(f"Track 94 features: {len(candidate_features[94])}")
print(f"Track 101 features: {len(candidate_features[101])}")

# Prepare aligned arrays for persistent storage.
source_frame_array, source_array = pack_features(source_features)
frames_94, features_94 = pack_features(candidate_features[94])
frames_101, features_101 = pack_features(candidate_features[101])

# Save features and frame numbers so future comparisons do not need
# to rerun YOLO, ByteTrack, or OSNet.
np.savez_compressed(
    OUTPUT_PATH,
    source_track=np.asarray(SOURCE_TRACK, dtype=np.int32),
    source_frames=source_frame_array,
    source_features=source_array,
    candidate_94_frames=frames_94,
    candidate_94_features=features_94,
    candidate_101_frames=frames_101,
    candidate_101_features=features_101,
)

print(f"\nSaved diagnostic features to: {OUTPUT_PATH}")

# Compare the extracted features in this run.
print_summary(
    "TRACK 6 -> TRACK 94",
    compare(source_features, candidate_features[94], matcher),
)

print_summary(
    "TRACK 6 -> TRACK 101",
    compare(source_features, candidate_features[101], matcher),
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
