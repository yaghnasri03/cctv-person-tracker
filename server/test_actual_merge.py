import ast
import numpy as np
from pathlib import Path


# Read video_processor.py without importing its model dependencies.
source_path = Path("video_processor.py")
source_code = source_path.read_text(encoding="utf-8")

tree = ast.parse(source_code)

# Extract only the PersonDetector class methods needed for merging.
class_node = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef)
    and node.name == "PersonDetector"
)

required_methods = {
    "normalize_feature",
    "cosine_similarity",
    "get_track_representative_feature",
    "merge_track_segments",
}

selected_methods = [
    node
    for node in class_node.body
    if isinstance(node, ast.FunctionDef)
    and node.name in required_methods
]

missing = required_methods - {
    node.name for node in selected_methods
}

if missing:
    raise RuntimeError(
        f"Required methods not found: {sorted(missing)}"
    )

# Create a temporary class containing the actual application methods.
test_class = ast.ClassDef(
    name="MergeTester",
    bases=[],
    keywords=[],
    body=selected_methods,
    decorator_list=[],
)

module = ast.Module(
    body=[test_class],
    type_ignores=[],
)

ast.fix_missing_locations(module)

namespace = {"np": np}
exec(compile(module, str(source_path), "exec"), namespace)

MergeTester = namespace["MergeTester"]
detector = MergeTester()

detector.MIN_OBSERVATIONS = 10
detector.MAX_TRACK_GAP = 90
detector.TRACK_MERGE_THRESHOLD = 0.80

# Load saved diagnostic features.
with np.load("transition_features.npz", allow_pickle=False) as saved:
    source_features = saved["source_features"]
    candidate_features = saved["candidate_101_features"]

# Synthetic frame records are used only to test the merge conditions.
track_data = {
    6: {
        "frames": list(range(488, 498)),
        "osnet_scores": [0.7],
        "osnet_features": list(source_features),
    },
    101: {
        "frames": list(range(536, 546)),
        "osnet_scores": [0.7],
        "osnet_features": list(candidate_features),
    },
}

print("ACTUAL MERGE METHOD DIAGNOSTIC")
print("-" * 45)
print("Loaded merge methods directly from video_processor.py.")
print("No model imports and no video processing.")
print()

groups = detector.merge_track_segments(track_data)

print()
print("RESULT")
print("-" * 45)

for group in groups:
    print(
        f"Tracks={group['track_ids']} | "
        f"First={group['first_frame']} | "
        f"Last={group['last_frame']}"
    )

print()
print("Reminder: synthetic frame records are used.")
print("This does not prove what happened in the original video run.")