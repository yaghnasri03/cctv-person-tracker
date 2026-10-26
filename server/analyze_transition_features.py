
import numpy as np

FEATURE_FILE = "transition_features.npz"


def normalize_rows(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def representative_similarity(features_a, features_b):
    # Average normalized embeddings, then normalize the resulting vectors.
    center_a = normalize_rows(features_a).mean(axis=0)
    center_b = normalize_rows(features_b).mean(axis=0)

    denominator = np.linalg.norm(center_a) * np.linalg.norm(center_b)

    if denominator == 0:
        return 0.0

    return float(np.dot(center_a, center_b) / denominator)


with np.load(FEATURE_FILE) as data:
    frames_6 = data["source_frames"]
    features_6 = data["source_features"]

    frames_94 = data["candidate_94_frames"]
    features_94 = data["candidate_94_features"]

    frames_101 = data["candidate_101_frames"]
    features_101 = data["candidate_101_features"]


print("Using saved features; no video inference will run.")
print(f"Track 6:   {len(features_6)} frames")
print(f"Track 94:  {len(features_94)} frames")
print(f"Track 101: {len(features_101)} frames")


for track_id, frames, features in [
    (94, frames_94, features_94),
    (101, frames_101, features_101),
]:
    print("\n" + "=" * 55)
    print(f"TRACK 6 -> TRACK {track_id}")
    print("=" * 55)

    if len(features_6) == 0 or len(features) == 0:
        print("Insufficient saved features.")
        continue

    pairwise = (
        normalize_rows(features_6)
        @ normalize_rows(features).T
    )

    print(f"Pairwise mean:          {pairwise.mean():.4f}")
    print(f"Pairwise median:        {np.median(pairwise):.4f}")
    print(f"Representative score:   "
          f"{representative_similarity(features_6, features):.4f}")

    print("\nCandidate frame averages:")

    for index, frame in enumerate(frames):
        print(
            f"Frame {frame}: "
            f"{pairwise[:, index].mean():.4f}"
        )

print("\nDone.")
