import cv2
from pathlib import Path

video_path = r"outputs\2328a724-013a-489a-8d73-80d7a773d2e0_processed.mp4"
output_dir = Path("production_inspection")
output_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise SystemExit("ERROR: Could not open processed video.")

fps = cap.get(cv2.CAP_PROP_FPS)

for seconds in [16, 16.5, 17, 17.5, 18, 18.5, 19]:
    frame_number = round(seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if success:
        output_path = output_dir / f"production_{seconds:.1f}s.jpg"
        cv2.imwrite(str(output_path), frame)
        print(f"Saved {output_path} (frame {frame_number})")
    else:
        print(f"Could not read frame {frame_number}")

cap.release()
print("\nExtraction complete. No detection models were loaded.")
