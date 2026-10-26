import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ==============================
# FILE PATHS
# ==============================

REFERENCE_IMAGE = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\sample frame matching.jpeg"
CCTV_FRAME = "test_frame.jpg"


# ==============================
# LOAD INSIGHTFACE
# ==============================

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


# ==============================
# READ IMAGES
# ==============================

print("\nReading reference image...")

reference_image = cv2.imread(REFERENCE_IMAGE)

if reference_image is None:
    print("❌ Could not read reference image.")
    exit()

print("Reading CCTV frame...")

cctv_image = cv2.imread(CCTV_FRAME)

if cctv_image is None:
    print("❌ Could not read CCTV frame.")
    exit()


# ==============================
# DETECT FACES
# ==============================

print("\nDetecting face in reference image...")

reference_faces = app.get(reference_image)

print("Reference faces detected:", len(reference_faces))

if len(reference_faces) == 0:
    print("❌ No face found in reference image.")
    exit()


print("\nDetecting faces in CCTV frame...")

cctv_faces = app.get(cctv_image)

print("CCTV faces detected:", len(cctv_faces))

if len(cctv_faces) == 0:
    print("❌ No faces found in CCTV frame.")
    exit()


# ==============================
# SELECT REFERENCE FACE
# ==============================

reference_face = max(
    reference_faces,
    key=lambda face: face.det_score
)

reference_embedding = reference_face.embedding

print("\n✅ Reference face selected.")


# ==============================
# COSINE SIMILARITY FUNCTION
# ==============================

def cosine_similarity(embedding1, embedding2):

    embedding1 = np.asarray(embedding1)
    embedding2 = np.asarray(embedding2)

    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) *
        np.linalg.norm(embedding2)
    )

    return float(similarity)


# ==============================
# COMPARE ALL CCTV FACES
# ==============================

print("\n")
print("======================================")
print("INSIGHTFACE FACE MATCHING RESULTS")
print("======================================")

results = []

for i, face in enumerate(cctv_faces, start=1):

    similarity = cosine_similarity(
        reference_embedding,
        face.embedding
    )

    bbox = face.bbox.astype(int).tolist()

    results.append(
        (i, similarity, bbox, face.det_score)
    )

    print(
        f"Face {i}: "
        f"Similarity = {similarity:.4f}, "
        f"Detection = {face.det_score:.4f}, "
        f"BBox = {bbox}"
    )


# ==============================
# FIND BEST MATCH
# ==============================

best_face = max(
    results,
    key=lambda x: x[1]
)

print("\n")
print("======================================")
print("BEST MATCH")
print("======================================")

print("Face number:", best_face[0])
print("Similarity:", f"{best_face[1]:.4f}")
print("Bounding box:", best_face[2])
print("Detection score:", f"{best_face[3]:.4f}")

print("======================================")

if best_face[1] >= 0.40:
    print("🟢 POSSIBLE MATCH")
else:
    print("🔴 NO STRONG MATCH")

print("======================================")