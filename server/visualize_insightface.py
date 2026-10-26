import cv2
import numpy as np
from insightface.app import FaceAnalysis


VIDEO_PATH = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\test video 2.mp4"

REFERENCE_IMAGE = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\sample frame matching.jpeg"

OUTPUT_IMAGE = "frame_1200_faces.jpg"


print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("✅ InsightFace loaded.")


# Read reference image
reference_image = cv2.imread(REFERENCE_IMAGE)

if reference_image is None:
    print("❌ Reference image not found.")
    exit()

reference_faces = app.get(reference_image)

if len(reference_faces) == 0:
    print("❌ No face found in reference image.")
    exit()

reference_face = max(
    reference_faces,
    key=lambda face: face.det_score
)

reference_embedding = reference_face.embedding


# Cosine similarity
def cosine_similarity(a, b):

    a = np.asarray(a)
    b = np.asarray(b)

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )


# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()


# Go to frame 1200
cap.set(cv2.CAP_PROP_POS_FRAMES, 1200)

ret, frame = cap.read()

cap.release()

if not ret:
    print("❌ Could not read frame 1200.")
    exit()


print("Running InsightFace on frame 1200...")

faces = app.get(frame)

print("Faces detected:", len(faces))


# Draw every face
for i, face in enumerate(faces, start=1):

    similarity = cosine_similarity(
        reference_embedding,
        face.embedding
    )

    x1, y1, x2, y2 = face.bbox.astype(int)

    # Draw bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Label
    label = f"Face {i} | Sim: {similarity:.3f}"

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    print(
        f"Face {i}: "
        f"Similarity={similarity:.4f}, "
        f"BBox={[x1, y1, x2, y2]}"
    )


# Save image
cv2.imwrite(
    OUTPUT_IMAGE,
    frame
)

print()
print("================================")
print("✅ IMAGE CREATED")
print("================================")
print("File:", OUTPUT_IMAGE)