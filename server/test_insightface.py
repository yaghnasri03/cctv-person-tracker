import cv2
from insightface.app import FaceAnalysis

IMAGE_PATH = "test_frame.jpg"

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

print("\nReading CCTV frame...")

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("❌ Could not read image.")
    exit()

print("Running face detection...")

faces = app.get(image)

print()
print("==============================")
print("Faces detected:", len(faces))
print("==============================")

for i, face in enumerate(faces, start=1):

    bbox = face.bbox.astype(int).tolist()

    print(
        f"Face {i}: "
        f"bbox={bbox}, "
        f"detection_score={face.det_score:.4f}"
    )