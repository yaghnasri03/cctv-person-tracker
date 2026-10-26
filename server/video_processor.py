import cv2
import time
import numpy as np

from ultralytics import YOLO
from person_matcher import PersonMatcher
from insightface.app import FaceAnalysis
from PIL import Image


class PersonDetector:

    # =========================================================
    # SETTINGS
    # =========================================================

    CONFIDENCE_THRESHOLD = 0.40
    FRAME_SKIP_INTERVAL = 3
    OSNET_INTERVAL = 15
    
    # OSNet is calculated every 15 frames
   

    # InsightFace is calculated every 30 frames
    FACE_INTERVAL = 30

    # Minimum observations for a valid track
    MIN_OBSERVATIONS = 10

    # Identity thresholds
    FACE_THRESHOLD = 0.40
    OSNET_THRESHOLD = 0.54

    # ---------------------------------------------------------
    # TRACK MERGING
    # ---------------------------------------------------------

    # Maximum gap allowed between two track segments
    MAX_TRACK_GAP = 90

    # Minimum OSNet similarity between two track segments
    # before considering them the same person
    TRACK_MERGE_THRESHOLD = 0.80

    def __init__(self):

        print("Loading YOLOv8 model...")

        self.model = YOLO("yolov8n.pt")

        print("✅ YOLOv8 model loaded.")

        print("Loading ONNX OSNet...")

        self.matcher = PersonMatcher()

        print("✅ ONNX OSNet loaded.")

        print("Loading InsightFace...")

        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.face_app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        print("✅ InsightFace loaded.")

    # =========================================================
    # COSINE SIMILARITY
    # =========================================================

    def cosine_similarity(
        self,
        feature1,
        feature2
    ):

        feature1 = np.asarray(
            feature1,
            dtype=np.float32
        ).reshape(-1)

        feature2 = np.asarray(
            feature2,
            dtype=np.float32
        ).reshape(-1)

        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(
            np.dot(feature1, feature2)
            / (norm1 * norm2)
        )

    # =========================================================
    # NORMALIZE FEATURE
    # =========================================================

    def normalize_feature(
        self,
        feature
    ):

        feature = np.asarray(
            feature,
            dtype=np.float32
        ).reshape(-1)

        norm = np.linalg.norm(
            feature
        )

        if norm == 0:
            return feature

        return feature / norm

    # =========================================================
    # EXTRACT REFERENCE FACE
    # =========================================================

    def extract_reference_face(
        self,
        reference_image_path
    ):

        image = cv2.imread(
            reference_image_path
        )

        if image is None:

            print(
                "❌ Could not read reference image."
            )

            return None

        faces = self.face_app.get(
            image
        )

        if not faces:

            print(
                "❌ No face found in reference image."
            )

            return None

        # Select the largest face
        best_face = max(
            faces,
            key=lambda face:
            (
                (face.bbox[2] - face.bbox[0])
                *
                (face.bbox[3] - face.bbox[1])
            )
        )

        if not hasattr(
            best_face,
            "embedding"
        ):

            print(
                "❌ Reference face embedding unavailable."
            )

            return None

        return np.asarray(
            best_face.embedding,
            dtype=np.float32
        )

    # =========================================================
    # FRAME TO TIMESTAMP
    # =========================================================

    def frame_to_timestamp(
        self,
        frame_number,
        fps
    ):

        if fps <= 0:
            fps = 30.0

        seconds = (
            frame_number / fps
        )

        hours = int(
            seconds // 3600
        )

        minutes = int(
            (seconds % 3600) // 60
        )

        remaining_seconds = (
            seconds % 60
        )

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{remaining_seconds:05.2f}"
        )

    # =========================================================
    # FACE SIMILARITY
    # =========================================================

    def get_face_similarity(
        self,
        person_crop,
        reference_face
    ):

        if person_crop is None:
            return 0.0

        if (
            person_crop.shape[0] < 40
            or person_crop.shape[1] < 20
        ):

            return 0.0

        try:

            faces = self.face_app.get(
                person_crop
            )

            if not faces:
                return 0.0

            best_similarity = 0.0

            for face in faces:

                if not hasattr(
                    face,
                    "embedding"
                ):
                    continue

                similarity = (
                    self.cosine_similarity(
                        face.embedding,
                        reference_face
                    )
                )

                if (
                    similarity
                    > best_similarity
                ):

                    best_similarity = (
                        similarity
                    )

            return best_similarity

        except Exception as e:

            print(
                f"Face processing warning: {e}"
            )

            return 0.0

    # =========================================================
    # GET REPRESENTATIVE OSNET FEATURE
    # =========================================================

    def get_track_representative_feature(
        self,
        data
    ):

        features = data.get(
            "osnet_features",
            []
        )

        if not features:
            return None

        normalized_features = []

        for feature in features:

            normalized = (
                self.normalize_feature(
                    feature
                )
            )

            if normalized.size > 0:

                normalized_features.append(
                    normalized
                )

        if not normalized_features:
            return None

        mean_feature = np.mean(
            normalized_features,
            axis=0
        )

        return self.normalize_feature(
            mean_feature
        )

    # =========================================================
    # MERGE TRACK SEGMENTS
    # =========================================================

    def merge_track_segments(
        self,
        track_data
    ):

        valid_tracks = []

        # -----------------------------------------------------
        # Prepare valid tracks
        # -----------------------------------------------------

        for track_id, data in (
            track_data.items()
        ):

            observations = len(
                data["frames"]
            )

            if (
                observations
                < self.MIN_OBSERVATIONS
            ):
                continue

            if not data[
                "osnet_scores"
            ]:
                continue

            representative = (
                self.get_track_representative_feature(
                    data
                )
            )

            if representative is None:
                continue

            valid_tracks.append({

                "track_id":
                    track_id,

                "first_frame":
                    min(data["frames"]),

                "last_frame":
                    max(data["frames"]),

                "representative":
                    representative
            })

        # -----------------------------------------------------
        # Sort tracks by starting frame
        # -----------------------------------------------------

        valid_tracks.sort(
            key=lambda x:
            x["first_frame"]
        )

        # -----------------------------------------------------
        # Start every track as its own group
        # -----------------------------------------------------

        groups = []

        for item in valid_tracks:

            groups.append({

                "track_ids": [
                    item["track_id"]
                ],

                "first_frame":
                    item["first_frame"],

                "last_frame":
                    item["last_frame"],

                "representative":
                    item["representative"]
            })

        # -----------------------------------------------------
        # Merge consecutive track segments
        # -----------------------------------------------------

        changed = True

        while changed:

            changed = False

            groups.sort(
                key=lambda x:
                x["first_frame"]
            )

            new_groups = []

            used = set()

            for i in range(
                len(groups)
            ):

                if i in used:
                    continue

                current = groups[i]

                merged_into_current = False

                for j in range(
                    i + 1,
                    len(groups)
                ):

                    if j in used:
                        continue

                    candidate = groups[j]

                    # -------------------------------------------------
                    # Candidate must start after current
                    # -------------------------------------------------

                    if (
                        candidate["first_frame"]
                        < current["first_frame"]
                    ):
                        continue

                    # -------------------------------------------------
                    # Calculate gap
                    # -------------------------------------------------

                    gap = (
                        candidate["first_frame"]
                        -
                        current["last_frame"]
                    )

                    # -------------------------------------------------
                    # IMPORTANT:
                    #
                    # If gap is negative, the two tracks overlap.
                    #
                    # They are therefore NOT consecutive track
                    # segments and MUST NOT be merged.
                    # -------------------------------------------------

                    if gap < 0:
                        continue

                    # -------------------------------------------------
                    # Gap too large
                    # -------------------------------------------------

                    if (
                        gap
                        > self.MAX_TRACK_GAP
                    ):
                        continue

                    # -------------------------------------------------
                    # Compare appearance
                    # -------------------------------------------------

                    appearance_similarity = (
                        self.cosine_similarity(
                            current[
                                "representative"
                            ],
                            candidate[
                                "representative"
                            ]
                        )
                    )

                    print(
                        f"Track segment comparison: "
                        f"{current['track_ids']} "
                        f"<-> "
                        f"{candidate['track_ids']} | "
                        f"Gap={gap} | "
                        f"OSNet={appearance_similarity:.4f}"
                    )

                    # -------------------------------------------------
                    # Merge only if appearance is sufficiently similar
                    # -------------------------------------------------

                    if (
                        appearance_similarity
                        >= self.TRACK_MERGE_THRESHOLD
                    ):

                        print(
    "Track segment comparison: {} <-> {} | Gap={} | OSNet={:.4f}".format(
        current["track_ids"],
        candidate["track_ids"],
        gap,
        appearance_similarity
    )
)
                        all_ids = (
                            current["track_ids"]
                            +
                            candidate["track_ids"]
                        )

                        # Combine representative features
                        representative = (
                            current[
                                "representative"
                            ]
                            +
                            candidate[
                                "representative"
                            ]
                        )

                        representative = (
                            self.normalize_feature(
                                representative
                            )
                        )

                        current = {

                            "track_ids":
                                all_ids,

                            "first_frame":
                                min(
                                    current[
                                        "first_frame"
                                    ],
                                    candidate[
                                        "first_frame"
                                    ]
                                ),

                            "last_frame":
                                max(
                                    current[
                                        "last_frame"
                                    ],
                                    candidate[
                                        "last_frame"
                                    ]
                                ),

                            "representative":
                                representative
                        }

                        used.add(j)

                        changed = True

                        merged_into_current = True

                        # Continue checking this expanded group
                        # against later segments.
                        continue

                new_groups.append(
                    current
                )

                used.add(i)

            groups = new_groups

        # -----------------------------------------------------
        # Print final groups
        # -----------------------------------------------------

        print()
        print("==============================")
        print("MERGED TRACK GROUPS")
        print("==============================")

        for i, group in enumerate(
            groups,
            start=1
        ):

            print(
                f"Group {i}: "
                f"Tracks={group['track_ids']} | "
                f"First={group['first_frame']} | "
                f"Last={group['last_frame']}"
            )

        return groups

    # =========================================================
    # MAIN VIDEO PROCESSING
    # =========================================================

    def process_video_for_detection(
        self,
        video_path,
        output_path,
        reference_image_path
    ):

        start_time = time.time()

        print()
        print("==============================")
        print("STARTING VIDEO PROCESSING")
        print("==============================")

        # =====================================================
        # REFERENCE OSNET
        # =====================================================

        print(
            "Extracting reference OSNet features..."
        )

        reference_osnet = (
            self.matcher.extract_features(
                reference_image_path
            )
        )

        print(
            "✅ Reference OSNet features extracted."
        )

        # =====================================================
        # REFERENCE FACE
        # =====================================================

        print(
            "Extracting reference face..."
        )

        reference_face = (
            self.extract_reference_face(
                reference_image_path
            )
        )

        if reference_face is None:

            return {
                "person_found": False,
                "match_count": 0,
                "timestamps": []
            }

        print(
            "✅ Reference face extracted."
        )

        # =====================================================
        # VIDEO INFORMATION
        # =====================================================

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            print(
                "❌ Could not open video."
            )

            return {
                "person_found": False,
                "match_count": 0,
                "timestamps": []
            }

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        cap.release()

        print(
            f"FPS: {fps}"
        )

        print(
            f"Resolution: "
            f"{width}x{height}"
        )

        print(
            f"Total frames: "
            f"{total_frames}"
        )

        # =====================================================
        # TRACK DATA
        # =====================================================

        track_data = {}

        all_track_boxes = {}

        # =====================================================
        # CONTINUOUS BYTE TRACK
        # =====================================================

        results = self.model.track(
            source=video_path,
            tracker="bytetrack.yaml",
            persist=True,
            stream=True,
            verbose=False
        )

        frame_number = 0

        for result in results:

            frame = result.orig_img

            if frame is None:

                frame_number += 1
                continue

            boxes = result.boxes

            if boxes is None:

                frame_number += 1
                continue

            if boxes.id is None:

                frame_number += 1
                continue

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

            confidences = (
                boxes.conf
                .cpu()
                .tolist()
            )

            class_ids = (
                boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            xyxy_boxes = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            # =================================================
            # PERSON TRACKS
            # =================================================

            for i, track_id in enumerate(
                track_ids
            ):

                confidence = (
                    confidences[i]
                )

                class_id = (
                    class_ids[i]
                )

                # COCO class 0 = person
                if class_id != 0:
                    continue

                if (
                    confidence
                    < self.CONFIDENCE_THRESHOLD
                ):
                    continue

                x1, y1, x2, y2 = (
                    xyxy_boxes[i]
                    .astype(int)
                )

                x1 = max(
                    0,
                    min(
                        x1,
                        width - 1
                    )
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        height - 1
                    )
                )

                x2 = max(
                    0,
                    min(
                        x2,
                        width
                    )
                )

                y2 = max(
                    0,
                    min(
                        y2,
                        height
                    )
                )

                if (
                    x2 <= x1
                    or
                    y2 <= y1
                ):
                    continue

                # =================================================
                # STORE TRACK BOX
                # =================================================

                if (
                    track_id
                    not in all_track_boxes
                ):

                    all_track_boxes[
                        track_id
                    ] = {}

                all_track_boxes[
                    track_id
                ][frame_number] = (
                    x1,
                    y1,
                    x2,
                    y2
                )

                # =================================================
                # INITIALIZE TRACK DATA
                # =================================================

                if (
                    track_id
                    not in track_data
                ):

                    track_data[
                        track_id
                    ] = {

                        "frames": [],

                        "boxes": [],

                        "osnet_scores": [],

                        "face_scores": [],

                        "osnet_features": [],

                        "strong_observations": 0
                    }

                data = track_data[
                    track_id
                ]

                data[
                    "frames"
                ].append(
                    frame_number
                )

                data[
                    "boxes"
                ].append(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

                # =================================================
                # PERSON CROP
                # =================================================

                person_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if person_crop.size == 0:
                    continue

                # =================================================
                # OSNET
                # =================================================

                osnet_similarity = None

                if (
                    frame_number
                    % self.OSNET_INTERVAL
                    == 0
                ):

                    try:

                        crop_rgb = (
                            cv2.cvtColor(
                                person_crop,
                                cv2.COLOR_BGR2RGB
                            )
                        )

                        crop_pil = (
                            Image.fromarray(
                                crop_rgb
                            )
                        )

                        person_features = (
                            self.matcher.extract_features(
                                crop_pil
                            )
                        )

                        osnet_similarity = (
                            self.matcher.compare_features(
                                reference_osnet,
                                person_features
                            )
                        )

                        data[
                            "osnet_scores"
                        ].append(
                            osnet_similarity
                        )

                        # Save feature for track merging
                        data[
                            "osnet_features"
                        ].append(
                            np.asarray(
                                person_features,
                                dtype=np.float32
                            ).reshape(-1)
                        )

                    except Exception as e:

                        print(
                            f"OSNet warning: {e}"
                        )

                # =================================================
                # INSIGHTFACE
                # =================================================

                face_similarity = None

                if (
                    frame_number
                    % self.FACE_INTERVAL
                    == 0
                ):

                    face_similarity = (
                        self.get_face_similarity(
                            person_crop,
                            reference_face
                        )
                    )

                    data[
                        "face_scores"
                    ].append(
                        face_similarity
                    )

                # =================================================
                # STRONG OBSERVATION
                # =================================================

                if (
                    face_similarity
                    is not None
                    and
                    osnet_similarity
                    is not None
                ):

                    if (
                        face_similarity
                        >= self.FACE_THRESHOLD
                        and
                        osnet_similarity
                        >= self.OSNET_THRESHOLD
                    ):

                        data[
                            "strong_observations"
                        ] += 1

            # =================================================
            # PROGRESS
            # =================================================

            frame_number += 1

            if (
                frame_number
                % 150
                == 0
            ):

                print(
                    f"Processing: "
                    f"{frame_number}/"
                    f"{total_frames} frames"
                )

        # =====================================================
        # MERGE TRACK SEGMENTS
        # =====================================================

        groups = (
            self.merge_track_segments(
                track_data
            )
        )

        # =====================================================
        # CREATE GROUP STATISTICS
        # =====================================================

        group_ranking = []

        for group_index, group in enumerate(
            groups
        ):

            combined = {

                "track_ids":
                    group["track_ids"],

                "frames": [],

                "boxes": [],

                "osnet_scores": [],

                "face_scores": [],

                "strong_observations": 0
            }

            # ---------------------------------------------
            # Combine all track segments
            # ---------------------------------------------

            for track_id in (
                group["track_ids"]
            ):

                if (
                    track_id
                    not in track_data
                ):
                    continue

                data = track_data[
                    track_id
                ]

                combined[
                    "frames"
                ].extend(
                    data["frames"]
                )

                combined[
                    "boxes"
                ].extend(
                    data["boxes"]
                )

                combined[
                    "osnet_scores"
                ].extend(
                    data["osnet_scores"]
                )

                combined[
                    "face_scores"
                ].extend(
                    data["face_scores"]
                )

                combined[
                    "strong_observations"
                ] += (
                    data[
                        "strong_observations"
                    ]
                )

            if not combined[
                "frames"
            ]:
                continue

            if not combined[
                "osnet_scores"
            ]:
                continue

            if not combined[
                "face_scores"
            ]:
                continue

            observations = len(
                combined["frames"]
            )

            first_frame = min(
                combined["frames"]
            )

            last_frame = max(
                combined["frames"]
            )

            best_face = max(
                combined["face_scores"]
            )

            average_face = (
                sum(
                    combined["face_scores"]
                )
                /
                len(
                    combined["face_scores"]
                )
            )

            best_osnet = max(
                combined["osnet_scores"]
            )

            average_osnet = (
                sum(
                    combined["osnet_scores"]
                )
                /
                len(
                    combined["osnet_scores"]
                )
            )

            coverage = (
                last_frame
                - first_frame
                + 1
            ) / total_frames

            # -------------------------------------------------
            # Identity-first ranking
            # -------------------------------------------------

            score = (

                0.60 * best_face

                + 0.20 * average_face

                + 0.15 * best_osnet

                + 0.05 * average_osnet
            )

            # Small temporal bonus
            score += (
                0.05 * coverage
            )

            group_ranking.append({

                "group_index":
                    group_index,

                "track_ids":
                    group[
                        "track_ids"
                    ],

                "score":
                    score,

                "observations":
                    observations,

                "first_frame":
                    first_frame,

                "last_frame":
                    last_frame,

                "coverage":
                    coverage,

                "best_face":
                    best_face,

                "average_face":
                    average_face,

                "best_osnet":
                    best_osnet,

                "average_osnet":
                    average_osnet,

                "strong_observations":
                    combined[
                        "strong_observations"
                    ]
            })

        # =====================================================
        # REMOVE VERY WEAK GROUPS
        # =====================================================

        group_ranking = [
            item
            for item in group_ranking
            if item["best_face"] >= 0.30
        ]

        # =====================================================
        # NO MATCH
        # =====================================================

        if not group_ranking:

            print()
            print(
                "❌ No sufficiently matching "
                "person group found."
            )

            total_time = (
                time.time()
                - start_time
            )

            print(
                f"Processing time: "
                f"{total_time:.2f} seconds"
            )

            return {
                "person_found": False,
                "match_count": 0,
                "timestamps": []
            }

        # =====================================================
        # SORT GROUPS
        # =====================================================

        group_ranking.sort(
            key=lambda x:
            x["score"],
            reverse=True
        )

        # =====================================================
        # PRINT GROUP RANKING
        # =====================================================

        print()
        print("==============================")
        print("IDENTITY GROUP RANKING")
        print("==============================")

        for i, item in enumerate(
            group_ranking,
            start=1
        ):

            print(
                f"{i}. "
                f"Tracks={item['track_ids']} | "
                f"Score={item['score']:.4f} | "
                f"Obs={item['observations']} | "
                f"First={item['first_frame']} | "
                f"Last={item['last_frame']} | "
                f"Coverage={item['coverage']:.2f} | "
                f"BestFace={item['best_face']:.4f} | "
                f"AvgFace={item['average_face']:.4f} | "
                f"BestOSNet={item['best_osnet']:.4f} | "
                f"AvgOSNet={item['average_osnet']:.4f} | "
                f"Strong={item['strong_observations']}"
            )

        # =====================================================
        # SELECT TARGET GROUP
        # =====================================================

        target = group_ranking[0]

        target_track_ids = (
            target["track_ids"]
        )

        print()
        print("==============================")
        print("TARGET GROUP IDENTIFIED")
        print("==============================")

        print(
            f"Target Track IDs: "
            f"{target_track_ids}"
        )

        print(
            f"Target Score: "
            f"{target['score']:.4f}"
        )

        print(
            f"Observations: "
            f"{target['observations']}"
        )

        print(
            f"Best Face: "
            f"{target['best_face']:.4f}"
        )

        print(
            f"Average Face: "
            f"{target['average_face']:.4f}"
        )

        print(
            f"Best OSNet: "
            f"{target['best_osnet']:.4f}"
        )

        print(
            f"Average OSNet: "
            f"{target['average_osnet']:.4f}"
        )

        print(
            f"Strong Observations: "
            f"{target['strong_observations']}"
        )

        print(
            f"First Frame: "
            f"{target['first_frame']}"
        )

        print(
            f"Last Frame: "
            f"{target['last_frame']}"
        )

        print(
            f"Coverage: "
            f"{target['coverage']:.2f}"
        )

        # =====================================================
        # CREATE FINAL TARGET BOXES
        # =====================================================

        target_boxes = {}

        for track_id in (
            target_track_ids
        ):

            if (
                track_id
                not in all_track_boxes
            ):
                continue

            for frame_no, box in (
                all_track_boxes[
                    track_id
                ].items()
            ):

                target_boxes[
                    frame_no
                ] = box

        # =====================================================
        # CREATE FINAL VIDEO
        # =====================================================

        print()
        print("==============================")
        print("CREATING FINAL VIDEO")
        print("==============================")

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            print(
                "❌ Could not reopen video."
            )

            return {
                "person_found": False,
                "match_count": 0,
                "timestamps": []
            }

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():

            cap.release()

            print(
                "❌ Could not create output video."
            )

            return {
                "person_found": False,
                "match_count": 0,
                "timestamps": []
            }

        target_frames = 0

        timestamps = []

        last_timestamp_frame = (
            -999999
        )

        timestamp_interval_frames = int(
            fps * 5
        )

        frame_index = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # =================================================
            # TARGET BOX
            # =================================================

            if (
                frame_index
                in target_boxes
            ):

                x1, y1, x2, y2 = (
                    target_boxes[
                        frame_index
                    ]
                )

                target_frames += 1

                # Green bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                # Target label
                cv2.putText(
                    frame,
                    "TARGET PERSON",
                    (
                        x1,
                        max(
                            30,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                timestamp_text = (
                    self.frame_to_timestamp(
                        frame_index,
                        fps
                    )
                )

                # Add timestamp every 5 seconds
                if (
                    frame_index
                    - last_timestamp_frame
                    >= timestamp_interval_frames
                ):

                    timestamps.append(
                        timestamp_text
                    )

                    last_timestamp_frame = (
                        frame_index
                    )

                # Timestamp on video
                cv2.putText(
                    frame,
                    (
                        "TARGET PERSON FOUND - "
                        + timestamp_text
                    ),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            writer.write(
                frame
            )

            frame_index += 1

        cap.release()
        writer.release()

        # =====================================================
        # LAST TARGET TIMESTAMP
        # =====================================================

        if target_frames > 0:

            last_target_frame = max(
                target_boxes.keys()
            )

            last_target_timestamp = (
                self.frame_to_timestamp(
                    last_target_frame,
                    fps
                )
            )

            if (
                not timestamps
                or
                timestamps[-1]
                != last_target_timestamp
            ):

                timestamps.append(
                    last_target_timestamp
                )

        # Remove duplicates
        timestamps = list(
            dict.fromkeys(
                timestamps
            )
        )

        # =====================================================
        # FINAL RESULTS
        # =====================================================

        person_found = (
            target_frames > 0
        )

        match_count = len(
            timestamps
        )

        print()
        print("==============================")
        print("FINAL RESULTS")
        print("==============================")

        print(
            f"Processed frames: "
            f"{frame_index}"
        )

        print(
            f"Target Track IDs: "
            f"{target_track_ids}"
        )

        print(
            f"Target frames: "
            f"{target_frames}"
        )

        print(
            f"Person found: "
            f"{person_found}"
        )

        print(
            f"Match count: "
            f"{match_count}"
        )

        print(
            f"Timestamps: "
            f"{timestamps}"
        )

        print(
            f"Output saved: "
            f"{output_path}"
        )

        # =====================================================
        # TOTAL PROCESSING TIME
        # =====================================================

        total_time = (
            time.time()
            - start_time
        )

        print()
        print("==============================")
        print("TOTAL PROCESSING TIME")
        print("==============================")

        print(
            f"{total_time:.2f} seconds"
        )

        print(
            f"{total_time / 60:.2f} minutes"
        )

        print("==============================")

        return {

            "person_found":
                person_found,

            "match_count":
                match_count,

            "timestamps":
                timestamps
        }