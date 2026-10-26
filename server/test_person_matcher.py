from person_matcher import PersonMatcher

# Replace this with the path to any person's image
image_path = "test.jpeg"

matcher = PersonMatcher()

features = matcher.extract_features(image_path)

print("Feature Shape:", features.shape)