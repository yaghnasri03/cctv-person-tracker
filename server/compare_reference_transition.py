import numpy as np
from pathlib import Path
from person_matcher import PersonMatcher

base = Path(r"uploads/2328a724-013a-489a-8d73-80d7a773d2e0")
reference_path = base / "matching pic.jpeg"
features_path = Path("transition_features.npz")

print("Loading OSNet...")
matcher = PersonMatcher()

print("Extracting reference appearance features...")
reference_features = matcher.extract_features(str(reference_path))

data = np.load(features_path)

source = data["source_features"]
candidate_94 = data["candidate_94_features"]
candidate_101 = data["candidate_101_features"]

def compare_saved(name, features):
    scores = [
        matcher.compare_features(reference_features, feature)
        for feature in features
    ]

    print(f"\n{name}")
    print(f"Observations: {len(scores)}")
    print(f"Mean similarity: {np.mean(scores):.4f}")
    print(f"Median similarity: {np.median(scores):.4f}")
    print(f"Best similarity: {np.max(scores):.4f}")

compare_saved("SOURCE TRACK 6 (frames 488-492)", source)
compare_saved("CANDIDATE TRACK 94 (frames 503-513)", candidate_94)
compare_saved("CANDIDATE TRACK 101 (frames 536-545)", candidate_101)

print("\nComparison complete. No video was loaded.")
