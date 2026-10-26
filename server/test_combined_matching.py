import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from person_matcher import PersonMatcher


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\test video 2.mp4"

REFERENCE_IMAGE = (
    r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150"
    r"\sample frame matching.jpeg"
)


# ============================================================
# SETTINGS
# ============================================================

REID_INTERVAL = 15
CONFIDENCE_THRESHOLD = 0.40


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("========================================")
print("Loading YOLOv8n...")
print("========================================")

yolo = YOLO("yolov8n.pt")

print("✅ YOLOv8n loaded.")


# ============================================================
# LOAD OSNET
# ============================================================

print()
print("========================================")
print("Loading OSNet...")
print("========================================")

osnet = PersonMatcher()

print("✅ OSNet loaded.")


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print()
print("========================================")
print("Loading InsightFace...")
print("========================================")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("✅ InsightFace loaded.")


# ============================================================
# FUNCTION: COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ============================================================
# EXTRACT REFERENCE FACE
# ============================================================

print()
print("========================================")
print("Extracting reference face...")
print("========================================")

reference_image = cv2.imread(
    REFERENCE_IMAGE
)

if reference_image is None:
    print("❌ Could not read reference image.")
    exit()


reference_faces = face_app.get(
    reference_image
)

print(
    "Reference faces detected:",
    len(reference_faces)
)


if len(reference_faces) == 0:
    print("❌ No face found in reference image.")
    exit()


# Choose the largest detected face
reference_face = max(
    reference_faces,
    key=lambda face: (
        face.bbox[2] - face.bbox[0]
    ) * (
        face.bbox[3] - face.bbox[1]
    )
)


reference_face_embedding = (
    reference_face.embedding
)


print(
    "Reference face detection score:",
    f"{reference_face.det_score:.4f}"
)

print("✅ Reference face extracted.")


# ============================================================
# EXTRACT REFERENCE OSNET
# ============================================================

print()
print("Extracting reference OSNet features...")

reference_osnet = osnet.extract_features(
    REFERENCE_IMAGE
)

print("✅ Reference OSNet features extracted.")


# ============================================================
# OPEN VIDEO
# ============================================================

print()
print("========================================")
print("Opening CCTV video...")
print("========================================")

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()


fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("FPS:", fps)
print("Total frames:", total_frames)
print(
    "Video duration:",
    f"{total_frames / fps:.2f} seconds"
)


# ============================================================
# TRACK DATA
# ============================================================

track_data = {}


frame_number = 0


print()
print("========================================")
print("Starting combined OSNet + InsightFace test")
print("========================================")

print(
    f"Re-ID interval: every {REID_INTERVAL} frames"
)


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # --------------------------------------------------------
    # YOLO + BYTETRACK
    # --------------------------------------------------------

    results = yolo.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )


    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            # ------------------------------------------------
            # PERSON ONLY
            # ------------------------------------------------

            class_id = int(
                box.cls.item()
            )

            if class_id != 0:
                continue


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = float(
                box.conf.item()
            )

            if confidence < CONFIDENCE_THRESHOLD:
                continue


            # ------------------------------------------------
            # TRACK ID
            # ------------------------------------------------

            if box.id is None:
                continue


            track_id = int(
                box.id.item()
            )


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )


            crop = frame[
                y1:y2,
                x1:x2
            ]


            if crop.size == 0:
                continue


            # ------------------------------------------------
            # CREATE TRACK RECORD
            # ------------------------------------------------

            if track_id not in track_data:

                track_data[track_id] = {
                    "osnet_scores": [],
                    "face_scores": [],
                    "last_checked": -REID_INTERVAL
                }


            current_track = track_data[
                track_id
            ]


            # ------------------------------------------------
            # RE-ID INTERVAL
            # ------------------------------------------------

            if (
                frame_number
                - current_track["last_checked"]
                < REID_INTERVAL
            ):
                continue


            current_track["last_checked"] = (
                frame_number
            )


            # =================================================
            # OSNET
            # =================================================

            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )

            person_image = Image.fromarray(
                crop_rgb
            )


            try:

                person_features = (
                    osnet.extract_features(
                        person_image
                    )
                )

                osnet_similarity = (
                    osnet.compare_features(
                        reference_osnet,
                        person_features
                    )
                )

            except Exception as e:

                print(
                    f"OSNet error Track {track_id}:",
                    e
                )

                continue


            current_track[
                "osnet_scores"
            ].append(
                osnet_similarity
            )


            # =================================================
            # INSIGHTFACE
            # =================================================

            try:

                faces = face_app.get(
                    crop
                )

            except Exception as e:

                print(
                    f"InsightFace error Track {track_id}:",
                    e
                )

                continue


            face_similarity = None


            if len(faces) > 0:

                # Select largest face
                selected_face = max(
                    faces,
                    key=lambda face: (
                        face.bbox[2] - face.bbox[0]
                    ) * (
                        face.bbox[3] - face.bbox[1]
                    )
                )


                face_similarity = cosine_similarity(
                    reference_face_embedding,
                    selected_face.embedding
                )


                current_track[
                    "face_scores"
                ].append(
                    face_similarity
                )


            # =================================================
            # PRINT RESULT
            # =================================================

            print(
                f"Frame {frame_number:4d} | "
                f"Track {track_id:3d} | "
                f"OSNet = {osnet_similarity:.4f} | "
                f"Face = "
                f"{face_similarity:.4f}"
                if face_similarity is not None
                else
                f"Frame {frame_number:4d} | "
                f"Track {track_id:3d} | "
                f"OSNet = {osnet_similarity:.4f} | "
                f"Face = N/A"
            )


# ============================================================
# CLOSE VIDEO
# ============================================================

cap.release()


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print()
print("======================================================")
print("              COMBINED MATCHING RESULTS")
print("======================================================")


for track_id, data in track_data.items():

    osnet_scores = data[
        "osnet_scores"
    ]

    face_scores = data[
        "face_scores"
    ]


    if len(osnet_scores) == 0:
        continue


    osnet_best = max(
        osnet_scores
    )

    osnet_average = sum(
        osnet_scores
    ) / len(osnet_scores)


    if len(face_scores) > 0:

        face_best = max(
            face_scores
        )

        face_average = sum(
            face_scores
        ) / len(face_scores)

    else:

        face_best = None
        face_average = None


    print()
    print(
        f"TRACK {track_id}"
    )

    print(
        "Observations:",
        len(osnet_scores)
    )

    print(
        f"OSNet  -> Best: {osnet_best:.4f} | "
        f"Average: {osnet_average:.4f}"
    )


    if face_best is not None:

        print(
            f"Face   -> Best: {face_best:.4f} | "
            f"Average: {face_average:.4f}"
        )

    else:

        print(
            "Face   -> No face observations"
        )


# ============================================================
# RANK TRACKS
# ============================================================

print()
print("======================================================")
print("          RANKED BY COMBINED EVIDENCE")
print("======================================================")


ranking = []


for track_id, data in track_data.items():

    osnet_scores = data[
        "osnet_scores"
    ]

    face_scores = data[
        "face_scores"
    ]


    if len(osnet_scores) == 0:
        continue


    osnet_best = max(
        osnet_scores
    )

    osnet_average = sum(
        osnet_scores
    ) / len(osnet_scores)


    if len(face_scores) > 0:

        face_best = max(
            face_scores
        )

        face_average = sum(
            face_scores
        ) / len(face_scores)

    else:

        face_best = 0.0
        face_average = 0.0


    # --------------------------------------------------------
    # COMBINED SCORE
    #
    # Face recognition is more useful for identity.
    # OSNet provides body/appearance evidence.
    # --------------------------------------------------------

    combined_score = (
        0.65 * face_average
        +
        0.35 * osnet_average
    )


    ranking.append(
        (
            track_id,
            combined_score,
            osnet_best,
            osnet_average,
            face_best,
            face_average,
            len(osnet_scores)
        )
    )


ranking.sort(
    key=lambda x: x[1],
    reverse=True
)


print()


for rank, item in enumerate(
    ranking,
    start=1
):

    (
        track_id,
        combined,
        osnet_best,
        osnet_avg,
        face_best,
        face_avg,
        observations
    ) = item


    print(
        f"{rank:2d}. "
        f"Track {track_id:3d} | "
        f"Combined={combined:.4f} | "
        f"OSNet Avg={osnet_avg:.4f} | "
        f"Face Avg={face_avg:.4f} | "
        f"Observations={observations}"
    )


print()
print("======================================================")
print("TEST COMPLETED")
print("======================================================")