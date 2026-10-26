from ultralytics import YOLO
import cv2
import numpy as np
import os


def test_yolo():
    print("\n=== Phase 1: AI Environment Setup ===")

    # Load YOLO model
    try:
        model = YOLO("yolov8n.pt")
        print("✅ YOLOv8n model loaded successfully.")
    except Exception as e:
        print("❌ Error loading YOLO model:", e)
        return

    # Create dummy image
    dummy_image = np.ones((640, 640, 3), dtype=np.uint8) * 255

    # Draw a rectangle (just for testing)
    cv2.rectangle(dummy_image, (220, 120), (420, 520), (0, 0, 0), -1)

    dummy_path = "dummy_person.jpg"
    cv2.imwrite(dummy_path, dummy_image)

    print("✅ Dummy image created.")

    try:
        results = model(dummy_path)

        print("✅ Inference completed successfully.")

        for r in results:
            print(f"Detected {len(r.boxes)} object(s).")

    except Exception as e:
        print("❌ Error during inference:", e)

    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
            print("✅ Dummy image deleted.")


if __name__ == "__main__":
    test_yolo()