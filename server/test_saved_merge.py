import numpy as np
from pathlib import Path

SOURCE_FILE = Path("transition_features.npz")
THRESHOLD = 0.80
MAX_GAP = 90
MIN_OBSERVATIONS = 10


def normalize_feature(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else feature


def representative_feature(features):
    normalized = [
        normalize_feature(feature)
        for feature in features
    ]
    return normalize_feature(np.mean(normalized, axis=0))


def cosine_similarity(feature1, feature2):
    feature1 = np.asarray(feature1, dtype=np.float32).reshape(-1)
    feature2 = np.asarray(feature2, dtype=np.float32).reshape(-1)

    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(feature1, feature2) / (norm1 * norm2))


with np.load(SOURCE_FILE, allow_pickle=False) as data:
    source_frames = data["source_frames"]
    source_features = data["source_features"]
    candidate_frames = data["candidate_101_frames"]
    candidate_features = data["candidate_101_features"]

    source_rep = representative_feature(source_features)
    candidate_rep = representative_feature(candidate_features)

    similarity = cosine_similarity(source_rep, candidate_rep)

    source_last = int(source_frames.max())
    candidate_first = int(candidate_frames.min())
    gap = candidate_first - source_last

    source_observations = len(source_frames)
    candidate_observations = len(candidate_frames)

    print("SAVED-FEATURE MERGE DIAGNOSTIC")
    print("-" * 40)
    print(f"Source: Track 6, saved features={source_observations}")
    print(f"Candidate: Track 101, saved features={candidate_observations}")
    print(f"Representative similarity: {similarity:.4f}")
    print(f"Merge threshold: {THRESHOLD:.2f}")
    print(f"Gap between saved feature ranges: {gap} frames")
    print(f"Maximum permitted gap: {MAX_GAP} frames")
    print()
    print("Similarity condition:", similarity >= THRESHOLD)
    print("Gap condition:", 0 <= gap <= MAX_GAP)
    print()
    print("IMPORTANT:")
    print("This uses selected saved features, not complete production tracks.")
    print("It does not prove the original tracks will merge.")
