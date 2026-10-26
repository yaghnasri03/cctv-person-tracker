
import numpy as np
from pathlib import Path

FILES = [
    Path("transition_features.npz"),
    Path("extra_candidate_features.npz"),
]

for path in FILES:
    print("\n" + "=" * 65)
    print(f"FILE: {path}")
    print("=" * 65)

    if not path.exists():
        print("File not found.")
        continue

    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            value = data[key]

            print(f"\n{key}")
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")

            if value.ndim == 1 and value.size <= 30:
                print(f"  Values: {value.tolist()}")
            elif key.endswith("_frames"):
                print(f"  Frame numbers: {value.tolist()}")

print("\nInspection complete. No video or models were loaded.")

