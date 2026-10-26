import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis


# ==========================================
# FILE PATHS
# ==========================================

VIDEO_PATH = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\test video 2.mp4"

REFERENCE_IMAGE = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\sample frame matching.jpeg"


# ==========================================
# SETTINGS
# ==========================================

REID_INTERVAL = 15

TARGET_FRAME = 1200

# This is Face 2 from our frame-1200 visualization.
TARGET_FACE_BBOX = [272, 78, 319, 137]


# ==========================================
# LOAD YOLO
# ==========================================

print("Loading YOLOv8n...")

yolo = YOLO("yolov8n.pt")

print("✅ YOLO loaded.")


# ==========================================
# LOAD INSIGHTFACE
# ==========================================

print("\nLoading InsightFace...")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("✅ InsightFace loaded.")


# ==========================================
# REFERENCE FACE
# ==========================================

print("\nReading reference image...")

reference_image = cv2.imread(
    REFERENCE_IMAGE
)

if reference_image is None:
    print("❌ Reference image not found.")
    exit()


reference_faces = face_app.get(
    reference_image
)

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

def cosine_similarity(a, b):

    a = np.asarray(a)
    b = np.asarray(b)

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ==========================================
# CHECK WHETHER FACE BELONGS TO PERSON
# ==========================================

def face_inside_person(
    face_bbox,
    person_bbox
):

    fx1, fy1, fx2, fy2 = face_bbox

    px1, py1, px2, py2 = person_bbox

    face_center_x = (
        fx1 + fx2
    ) / 2

    face_center_y = (
        fy1 + fy2
    ) / 2

    return (
        px1 <= face_center_x <= px2
        and
        py1 <= face_center_y <= py2
    )


# ==========================================
# FIND CLOSEST FACE TO TARGET FACE BOX
# ==========================================

def bbox_center_distance(
    bbox1,
    bbox2
):

    x1 = (
        bbox1[0] +
        bbox1[2]
    ) / 2

    y1 = (
        bbox1[1] +
        bbox1[3]
    ) / 2

    x2 = (
        bbox2[0] +
        bbox2[2]
    ) / 2

    y2 = (
        bbox2[1] +
        bbox2[3]
    ) / 2

    return (
        (x1 - x2) ** 2
        +
        (y1 - y2) ** 2
    )


# ==========================================
# OPEN VIDEO
# ==========================================

print("\nOpening CCTV video...")

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    print("❌ Could not open video.")
    exit()


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

print(
    "Total frames:",
    total_frames
)

print(
    "FPS:",
    fps
)


# ==========================================
# TRACK DATA
# ==========================================

track_data = {}


# ==========================================
# TARGET TRACK ID
# ==========================================

target_track_id = None


# ==========================================
# SIMILARITY HISTORY
# ==========================================

target_similarities = []


# ==========================================
# PROCESS VIDEO FROM START TO END
# ==========================================

print("\n")
print("==========================================")
print("PROCESSING VIDEO")
print("==========================================")


frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break


    # --------------------------------------
    # BYTE TRACK
    # --------------------------------------

    results = yolo.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )


    # --------------------------------------
    # ONLY RUN INSIGHTFACE EVERY 15 FRAMES
    # --------------------------------------

    if frame_number % REID_INTERVAL != 0:

        frame_number += 1
        continue


    # --------------------------------------
    # COLLECT PERSON TRACKS
    # --------------------------------------

    person_tracks = []


    for result in results:

        if result.boxes is None:
            continue

        if result.boxes.id is None:
            continue


        for i, box in enumerate(
            result.boxes
        ):

            class_id = int(
                box.cls[0]
            )

            if class_id != 0:
                continue


            confidence = float(
                box.conf[0]
            )

            if confidence < 0.40:
                continue


            track_id = int(
                result.boxes.id[i]
            )


            person_bbox = [
                int(x)
                for x in box.xyxy[0].tolist()
            ]


            person_tracks.append(
                (
                    track_id,
                    person_bbox
                )
            )


    # --------------------------------------
    # DETECT FACES
    # --------------------------------------

    faces = face_app.get(
        frame
    )


    # --------------------------------------
    # MATCH EACH FACE TO PERSON TRACK
    # --------------------------------------

    for face in faces:

        face_bbox = [
            int(x)
            for x in face.bbox
        ]


        # Find which person contains face
        matching_track = None


        for track_id, person_bbox in person_tracks:

            if face_inside_person(
                face_bbox,
                person_bbox
            ):

                matching_track = track_id

                break


        if matching_track is None:

            continue


        # ----------------------------------
        # FACE SIMILARITY
        # ----------------------------------

        similarity = cosine_similarity(
            reference_embedding,
            face.embedding
        )


        # ----------------------------------
        # SAVE TRACK DATA
        # ----------------------------------

        if matching_track not in track_data:

            track_data[matching_track] = []


        track_data[
            matching_track
        ].append(
            (
                frame_number,
                similarity
            )
        )


        # ----------------------------------
        # FRAME 1200 TARGET IDENTIFICATION
        # ----------------------------------

        if frame_number == TARGET_FRAME:

            distance = bbox_center_distance(
                face_bbox,
                TARGET_FACE_BBOX
            )


            # Keep closest face to our known Face 2
            if (
                target_track_id is None
                or
                distance < target_face_distance
            ):

                target_track_id = matching_track

                target_face_distance = distance


    frame_number += 1


# ==========================================
# CLOSE VIDEO
# ==========================================

cap.release()


# ==========================================
# RESULTS
# ==========================================

print("\n")
print("==========================================")
print("TRACKING COMPLETE")
print("==========================================")


print(
    "Unique Track IDs:",
    len(track_data)
)


print(
    "Target Track ID:",
    target_track_id
)


# ==========================================
# TARGET HISTORY
# ==========================================

if target_track_id is None:

    print(
        "❌ Could not identify target Track ID."
    )

    exit()


target_similarities = track_data.get(
    target_track_id,
    []
)


print(
    "Target observations:",
    len(target_similarities)
)


if len(target_similarities) == 0:

    print(
        "❌ No similarity observations found."
    )

    exit()


# ==========================================
# STATISTICS
# ==========================================

values = [
    similarity
    for _, similarity
    in target_similarities
]


print("\n")
print("==========================================")
print("TARGET FACE SIMILARITY")
print("==========================================")


print(
    "Best:",
    f"{max(values):.4f}"
)

print(
    "Average:",
    f"{np.mean(values):.4f}"
)

print(
    "Minimum:",
    f"{min(values):.4f}"
)

print(
    "Median:",
    f"{np.median(values):.4f}"
)


# ==========================================
# TOP MATCHES
# ==========================================

print("\nTop 15 target similarities:")


top_values = sorted(
    target_similarities,
    key=lambda x: x[1],
    reverse=True
)[:15]


for frame_number, similarity in top_values:

    timestamp = (
        frame_number / fps
    )

    print(
        f"Frame {frame_number:4d} "
        f"| Time {timestamp:6.2f}s "
        f"| Similarity {similarity:.4f}"
    )


print("\n")
print("==========================================")
print("✅ TEST COMPLETED")
print("==========================================")