
import numpy as np

BASELINE_FILE = "transition_features.npz"
EXTRA_FILE = "extra_candidate_features.npz"


def normalize_rows(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def representative_similarity(source, candidate):
    source_center = normalize_rows(source).mean(axis=0)
    candidate_center = normalize_rows(candidate).mean(axis=0)

    denominator = (
        np.linalg.norm(source_center)
        * np.linalg.norm(candidate_center)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(source_center, candidate_center) / denominator
    )


def compare(source, candidate):
    source_norm = normalize_rows(source)
    candidate_norm = normalize_rows(candidate)

    matrix = source_norm @ candidate_norm.T

    return {
        "mean": float(matrix.mean()),
        "median": float(np.median(matrix)),
        "maximum": float(matrix.max()),
        "minimum": float(matrix.min()),
        "representative": representative_similarity(
            source, candidate
        ),
        "frame_means": matrix.mean(axis=0),
    }


with np.load(BASELINE_FILE) as baseline:
    source_frames = baseline["source_frames"]
    source_features = baseline["source_features"]

    candidates = {
        94: (
            baseline["candidate_94_frames"],
            baseline["candidate_94_features"],
        ),
        101: (
            baseline["candidate_101_frames"],
            baseline["candidate_101_features"],
        ),
    }

with np.load(EXTRA_FILE) as extra:
    for track_id in (3, 99):
        candidates[track_id] = (
            extra[f"track_{track_id}_frames"],
            extra[f"track_{track_id}_features"],
        )

print("Saved-feature comparison only; no video inference.")
print(f"Track 6 source frames: {source_frames.tolist()}")

results = {}

for track_id, (frames, features) in candidates.items():
    print("\n" + "=" * 55)
    print(f"TRACK 6 -> TRACK {track_id}")
    print("=" * 55)

    if len(features) == 0:
        print("No saved features available.")
        continue

    result = compare(source_features, features)
    results[track_id] = result

    print(f"Candidate frames:       {frames.tolist()}")
    print(f"Pairwise mean:          {result['mean']:.4f}")
    print(f"Pairwise median:        {result['median']:.4f}")
    print(f"Pairwise maximum:       {result['maximum']:.4f}")
    print(f"Pairwise minimum:       {result['minimum']:.4f}")
    print(f"Representative score:   {result['representative']:.4f}")

    print("\nPer-frame mean similarity:")
    for frame, score in zip(frames, result["frame_means"]):
        print(f"Frame {frame}: {score:.4f}")

print("\n" + "=" * 55)
print("CANDIDATE RANKING")
print("=" * 55)

ranking = sorted(
    results.items(),
    key=lambda item: item[1]["representative"],
    reverse=True,
)

for rank, (track_id, result) in enumerate(ranking, start=1):
    print(
        f"{rank}. Track {track_id}: "
        f"representative={result['representative']:.4f}, "
        f"pairwise_mean={result['mean']:.4f}"
    )

print("\nAnalysis complete.")
