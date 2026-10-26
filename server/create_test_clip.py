import cv2

input_video = r"D:\epics project\server\test video 2.mp4"
output_video = r"D:\epics project\server\test_clip_15sec.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Could not open the original video.")
    print("Check the input_video path.")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0 or width <= 0 or height <= 0:
    print("ERROR: Invalid video properties.")
    cap.release()
    raise SystemExit(1)

total_frames = int(fps * 15)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

if not out.isOpened():
    print("ERROR: Could not create the output video.")
    cap.release()
    raise SystemExit(1)

count = 0

while count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    count += 1

cap.release()
out.release()

print("\nSUCCESS!")
print(f"Output: {output_video}")
print(f"Frames: {count}")
print(f"Duration: {count / fps:.2f} seconds")
