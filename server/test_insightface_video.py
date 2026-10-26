import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ==========================================
# FILE PATHS
# ==========================================

VIDEO_PATH = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\test video 2.mp4"

REFERENCE_IMAGE = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\sample frame matching.jpeg"


# ==========================================
# LOAD INSIGHTFACE
# ==========================================

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


# ==========================================
# READ REFERENCE IMAGE
# ==========================================

print("\nReading reference image...")

reference_image = cv2.imread(REFERENCE_IMAGE)

if reference_image is None:
    print("❌ Could not read reference image.")
    exit()

reference_faces = app.get(reference_image)

print("Reference faces:", len(reference_faces))

if len(reference_faces) == 0:
    print("❌ No face found in reference image.")
    exit()

reference_face = max(
    reference_faces,
    key=lambda face: face.det_score
)

reference_embedding = reference_face.embedding

print("✅ Reference face extracted.")


# ==========================================
# COSINE SIMILARITY
# ==========================================

def cosine_similarity(embedding1, embedding2):

    embedding1 = np.asarray(embedding1)
    embedding2 = np.asarray(embedding2)

    return float(
        np.dot(embedding1, embedding2)
        /
        (
            np.linalg.norm(embedding1)
            *
            np.linalg.norm(embedding2)
        )
    )


# ==========================================
# OPEN VIDEO
# ==========================================

print("\nOpening CCTV video...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Total frames:", total_frames)
print("FPS:", fps)

# Test approximately every 300 frames
test_frames = [
    0,
    300,
    600,
    900,
    1200
]

# Make sure we don't request frames beyond the video
test_frames = [
    frame_number
    for frame_number in test_frames
    if frame_number < total_frames
]


# ==========================================
# TEST MULTIPLE FRAMES
# ==========================================

print("\n")
print("==============================================")
print("TESTING INSIGHTFACE THROUGHOUT THE VIDEO")
print("==============================================")


overall_best_similarity = -1
overall_best_frame = None
overall_best_face = None


for frame_number in test_frames:

    print(f"\n--- Testing frame {frame_number} ---")

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    ret, frame = cap.read()

    if not ret:
        print("❌ Could not read frame.")
        continue

    faces = app.get(frame)

    print("Faces detected:", len(faces))

    if len(faces) == 0:
        print("No faces found.")
        continue

    frame_best_similarity = -1
    frame_best_face_number = None

    for face_number, face in enumerate(faces, start=1):

        similarity = cosine_similarity(
            reference_embedding,
            face.embedding
        )

        bbox = face.bbox.astype(int).tolist()

        print(
            f"Face {face_number}: "
            f"Similarity = {similarity:.4f}, "
            f"Detection = {face.det_score:.4f}, "
            f"BBox = {bbox}"
        )

        if similarity > frame_best_similarity:

            frame_best_similarity = similarity
            frame_best_face_number = face_number

        if similarity > overall_best_similarity:

            overall_best_similarity = similarity
            overall_best_frame = frame_number
            overall_best_face = face_number

    print(
        f"Best face in frame {frame_number}: "
        f"Face {frame_best_face_number} "
        f"with similarity {frame_best_similarity:.4f}"
    )


# ==========================================
# FINAL RESULT
# ==========================================

cap.release()

print("\n")
print("==============================================")
print("OVERALL BEST MATCH")
print("==============================================")

print("Frame:", overall_best_frame)
print("Face:", overall_best_face)
print(
    "Similarity:",
    f"{overall_best_similarity:.4f}"
)

print("==============================================")

if overall_best_similarity >= 0.40:
    print("🟢 POSSIBLE FACE MATCH")
else:
    print("🔴 NO STRONG FACE MATCH")

print("==============================================")