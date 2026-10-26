from person_matcher import PersonMatcher
import os

REFERENCE_IMAGE = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\sample frame matching.jpeg"
CROP_DIR = "person_crops"

print("Loading PersonMatcher...")
matcher = PersonMatcher()

print("\nExtracting reference image features...")
reference_features = matcher.extract_features(REFERENCE_IMAGE)

print("\n==============================")
print("Comparing reference with crops")
print("==============================")

for filename in sorted(os.listdir(CROP_DIR)):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    crop_path = os.path.join(CROP_DIR, filename)

    print(f"\nTesting: {filename}")

    crop_features = matcher.extract_features(crop_path)

    similarity = matcher.compare_features(
        reference_features,
        crop_features
    )

    print(f"Similarity: {similarity:.4f}")

print("\n==============================")
print("Test completed.")
print("==============================")