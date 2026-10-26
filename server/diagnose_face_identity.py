import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from insightface.app import FaceAnalysis

VIDEO_PATH = Path(
    r"outputs\2328a724-013a-489a-8d73-80d7a773d2e0_processed.mp4"
)

REFERENCE_PATH = Path(
    r"uploads\2328a724-013a-489a-8d73-80d7a773d2e0\matching pic.jpeg"
)

OUTPUT_DIR = Path("face_identity_diagnostic")
OUTPUT_DIR.mkdir(exist_ok=True)

TIMES = [18.0, 19.0]


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def main():
    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        return

    if not REFERENCE_PATH.exists():
        print(f"Reference image not found: {REFERENCE_PATH}")
        return

    print("Loading InsightFace...")

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    face_app.prepare(ctx_id=-1, det_size=(640, 640))

    print("Reading reference image...")

    reference_image = cv2.imread(str(REFERENCE_PATH))

    if reference_image is None:
        print("Could not read reference image.")
        return

    reference_faces = face_app.get(reference_image)

    if not reference_faces:
        print("No face detected in the reference image.")
        return

    reference_face = max(
        reference_faces,
        key=lambda face: (
            (face.bbox[2] - face.bbox[0])
            * (face.bbox[3] - face.bbox[1])
        )
    )

    reference_embedding = reference_face.embedding

    print("Reference face extracted.")
    print("Loading YOLO...")

    detector = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Could not open processed video.")
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

                faces = face_app.get(crop)

                face_scores = []

                for face in faces:
                    score = cosine_similarity(
                        face.embedding,
                        reference_embedding
                    )
                    face_scores.append(score)

                # A missing face is not evidence of a mismatch.
                best_score = (
                    max(face_scores)
                    if face_scores
                    else None
                )

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "score": best_score,
                    "face_count": len(faces)
                })

        detections.sort(
            key=lambda item: (
                item["score"]
                if item["score"] is not None
                else -1
            ),
            reverse=True
        )

        annotated = frame.copy()

        for rank, item in enumerate(detections, start=1):
            x1, y1, x2, y2 = item["box"]
            score = item["score"]

            if score is None:
                label = f"Rank {rank}: no face"
            else:
                label = f"Rank {rank}: face {score:.3f}"

            print(
                f"Person {rank}: {label}, "
                f"faces={item['face_count']}, "
                f"box=({x1}, {y1}, {x2}, {y2})"
            )

            color = (
                (0, 255, 0)
                if rank == 1 and score is not None
                else (0, 165, 255)
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1
            )

        output_path = OUTPUT_DIR / (
            f"face_diagnostic_{seconds:.1f}s.jpg"
        )

        cv2.imwrite(str(output_path), annotated)

        print(f"Saved image: {output_path}")

    cap.release()
    print("\nFace diagnostic completed.")


if __name__ == "__main__":
    main()