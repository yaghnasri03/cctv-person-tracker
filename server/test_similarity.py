from person_matcher import PersonMatcher

matcher = PersonMatcher()

feature1 = matcher.extract_features("test.jpeg")
feature2 = matcher.extract_features("test.jpeg")

similarity = matcher.compare_features(feature1, feature2)

print("Similarity:", similarity)