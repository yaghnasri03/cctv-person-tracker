import cv2
from insightface.app import FaceAnalysis


REFERENCE_IMAGE = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\sample frame matching.jpeg"


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


image = cv2.imread(REFERENCE_IMAGE)

if image is None:
    print("❌ Could not read reference image.")
    exit()


faces = app.get(image)

print("Faces detected:", len(faces))

if len(faces) == 0:
    print("❌ No face detected.")
    exit()


face = max(
    faces,
    key=lambda f: f.det_score
)

x1, y1, x2, y2 = face.bbox.astype(int)

print("Face bounding box:")
print(x1, y1, x2, y2)

print("Detection score:", face.det_score)


# Add some padding around the face
padding = 30

x1 = max(0, x1 - padding)
y1 = max(0, y1 - padding)
x2 = min(image.shape[1], x2 + padding)
y2 = min(image.shape[0], y2 + padding)

face_crop = image[y1:y2, x1:x2]

cv2.imwrite(
    "reference_face_crop.jpg",
    face_crop
)

print()
print("================================")
print("✅ REFERENCE FACE EXTRACTED")
print("================================")
print("Created: reference_face_crop.jpg")
print("================================")