import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from person_matcher import PersonMatcher

VIDEO_PATH = Path(
    r"outputs\2328a724-013a-489a-8d73-80d7a773d2e0_processed.mp4"
)

REFERENCE_PATH = Path(
    r"uploads\2328a724-013a-489a-8d73-80d7a773d2e0\matching pic.jpeg"
)

OUTPUT_DIR = Path("identity_diagnostic")
OUTPUT_DIR.mkdir(exist_ok=True)

TIMES = [18.0, 19.0]


def main():
    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        return

    if not REFERENCE_PATH.exists():
        print(f"Reference image not found: {REFERENCE_PATH}")
        return

    print("Loading YOLO...")
    detector = YOLO("yolov8n.pt")

    print("Loading OSNet...")
    matcher = PersonMatcher()

    reference_image = Image.open(
        REFERENCE_PATH
    ).convert("RGB")

    reference_features = matcher.extract_features(
        reference_image
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Could not open the processed video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps:.3f}")

    for seconds in TIMES:
        frame_number = round(seconds * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            print(f"Could not read frame at {seconds}s")
            continue

        print(f"\n{'=' * 55}")
        print(f"FRAME AT {seconds:.1f} SECONDS")
        print(f"{'=' * 55}")

        result = detector.predict(
            source=frame,
            classes=[0],
            conf=0.25,
            verbose=False
        )[0]

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].cpu().numpy()
                )

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # OSNet preprocessing in PersonMatcher converts
                # the supplied PIL image to RGB.
                crop_rgb = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2RGB
                )
                crop_image = Image.fromarray(crop_rgb)

                features = matcher.extract_features(
                    crop_image
                )

                similarity = matcher.compare_features(
                    reference_features,
                    features
                )

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "similarity": similarity,
                    "crop": crop
                })

        detections.sort(
            key=lambda item: item["similarity"],
            reverse=True
        )

        annotated = frame.copy()

        for rank, item in enumerate(detections, start=1):
            x1, y1, x2, y2 = item["box"]
            score = item["similarity"]

            print(
                f"Person {rank}: "
                f"OSNet similarity={score:.4f}, "
                f"box=({x1}, {y1}, {x2}, {y2})"
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0) if rank == 1 else (0, 165, 255),
                2
            )

            cv2.putText(
                annotated,
                f"Rank {rank}: {score:.3f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0) if rank == 1 else (0, 165, 255),
                2
            )

        output_path = OUTPUT_DIR / (
            f"diagnostic_{seconds:.1f}s.jpg"
        )

        cv2.imwrite(
            str(output_path),
            annotated
        )

        print(f"Saved annotated frame: {output_path}")

    cap.release()
    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()