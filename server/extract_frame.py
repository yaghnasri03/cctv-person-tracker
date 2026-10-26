import cv2

VIDEO_PATH = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\test video 2.mp4"
OUTPUT_PATH = "test_frame.jpg"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

ret, frame = cap.read()

if not ret:
    print("❌ Could not read first frame.")
    cap.release()
    exit()

cv2.imwrite(OUTPUT_PATH, frame)

cap.release()

print("✅ Frame extracted successfully.")
print("Created:", OUTPUT_PATH)