import cv2
from ultralytics import YOLO
import os

VIDEO_PATH = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\test video 2.mp4"
OUTPUT_DIR = "person_crops"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading YOLOv8n...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Could not read first frame.")
    exit()

print("Running YOLO detection...")

results = model(
    frame,
    verbose=False
)

person_count = 0

for result in results:

    if result.boxes is None:
        continue

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # COCO class 0 = person
        if class_id != 0:
            continue

        if confidence < 0.40:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        person_count += 1

        filename = os.path.join(
            OUTPUT_DIR,
            f"person_{person_count}.jpg"
        )

        cv2.imwrite(filename, crop)

        print(
            f"Person {person_count}: "
            f"confidence={confidence:.3f}, "
            f"box=({x1},{y1},{x2},{y2})"
        )

print()
print("================================")
print("Persons detected:", person_count)
print("Crops saved in:", OUTPUT_DIR)
print("================================")