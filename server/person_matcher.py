import numpy as np
import onnxruntime as ort

from PIL import Image
from torchvision import transforms


class PersonMatcher:
    def __init__(self):
        print("Loading Person Re-Identification model...")

        self.model_path = "osnet_x1_0.onnx"

        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        print("✅ OSNet ONNX model loaded.")
        print("✅ Execution Provider: CPUExecutionProvider")

    def extract_features(self, image_input):

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        image = self.transform(image)

        image = image.unsqueeze(0).numpy().astype(np.float32)

        features = self.session.run(
            [self.output_name],
            {self.input_name: image}
        )[0]

        return features

    def compare_features(self, feature1, feature2):

        feature1 = np.asarray(feature1, dtype=np.float32)
        feature2 = np.asarray(feature2, dtype=np.float32)

        feature1 = feature1.reshape(-1)
        feature2 = feature2.reshape(-1)

        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(feature1, feature2) / (norm1 * norm2)

        return float(similarity)