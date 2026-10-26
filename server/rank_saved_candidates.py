
import numpy as np
from pathlib import Path

SOURCE_FILE = Path("transition_features.npz")
EXTRA_FILE = Path("extra_candidate_features.npz")


def normalize_rows(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def representative(features):
    normalized = normalize_rows(features)
    mean_vector = normalized.mean(axis=0)
    norm = np.linalg.norm(mean_vector)
    if norm < 1e-12:
        return mean_vector
    return mean_vector / norm


def compare(source_features, candidate_features):
    source = normalize_rows(source_features)
    candidate = normalize_rows(candidate_features)
    pairwise = source @ candidate.T

    source_rep = representative(source_features)
    candidate_rep = representative(candidate_features)

    return {
        "pairwise_mean": float(pairwise.mean()),
        "pairwise_median": float(np.median(pairwise)),
        "pairwise_max": float(pairwise.max()),
        "representative": float(source_rep @ candidate_rep),
    }


if not SOURCE_FILE.exists() or not EXTRA_FILE.exists():
    raise FileNotFoundError(
        "Both transition_features.npz and "
        "extra_candidate_features.npz must exist."
    )

with np.load(SOURCE_FILE, allow_pickle=False) as base, \
     np.load(EXTRA_FILE, allow_pickle=False) as extra:

    source = base["source_features"]

    candidates = {
        "Track 94": base["candidate_94_features"],
        "Track 101": base["candidate_101_features"],
        "Track 3": extra["track_3_features"],
        "Track 99": extra["track_99_features"],
    }

    print("Saved-feature comparison only; no inference performed.")
    print(f"Source Track 6 feature count: {len(source)}")

    results = []

    for name, features in candidates.items():
        if len(source) == 0 or len(features) == 0:
            print(f"Skipping {name}: missing features")
            continue

        scores = compare(source, features)
        results.append((name, scores))

    results.sort(
        key=lambda item: item[1]["representative"],
        reverse=True,
    )

    print("\nCANDIDATE RANKING")
    print("-" * 80)

    for rank, (name, scores) in enumerate(results, start=1):
        print(
            f"{rank}. {name:10s} | "
            f"Pairwise mean={scores['pairwise_mean']:.4f} | "
            f"Median={scores['pairwise_median']:.4f} | "
            f"Max={scores['pairwise_max']:.4f} | "
            f"Representative={scores['representative']:.4f}"
        )

print("\nDone. No video or models were loaded.")
