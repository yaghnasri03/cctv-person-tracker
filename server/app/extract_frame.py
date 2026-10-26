import cv2

VIDEO_PATH = "test video 2.mp4"
OUTPUT_PATH = "test_frame.jpg"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

# Read the first frame
ret, frame = cap.read()

if not ret:
    print("❌ Could not read first frame.")
    cap.release()
    exit()

cv2.imwrite(
    OUTPUT_PATH,
    frame
)

cap.release()

print("✅ Frame extracted successfully.")
print("Created:", OUTPUT_PATH)