from ultralytics import YOLO

print("Loading YOLOv8n...")

model = YOLO("yolov8n.pt")

print("Testing ByteTrack...")

results = model.track(
    source="test_video.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    verbose=False
)

print("✅ ByteTrack test completed.")

for result in results:
    if result.boxes is not None and result.boxes.id is not None:
        print("Track IDs:", result.boxes.id.int().tolist())