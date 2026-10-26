from person_matcher import PersonMatcher

REFERENCE_IMAGE = r"uploads\59f85436-beec-4c1e-bafc-ff0366d2b150\sample frame matching.jpeg"
CCTV_FRAME = "test_frame.jpg"

print("Loading PersonMatcher...")
matcher = PersonMatcher()

print("\nExtracting reference image features...")
reference_features = matcher.extract_features(REFERENCE_IMAGE)

print("Extracting CCTV frame features...")
frame_features = matcher.extract_features(CCTV_FRAME)

print("\nComparing...")
similarity = matcher.compare_features(
    reference_features,
    frame_features
)

print("\n==============================")
print("REFERENCE:", REFERENCE_IMAGE)
print("CCTV FRAME:", CCTV_FRAME)
print("OSNet Similarity:", similarity)
print("==============================")

if similarity >= 0.75:
    print("🟢 TARGET MATCH")
else:
    print("🔴 NOT A MATCH")