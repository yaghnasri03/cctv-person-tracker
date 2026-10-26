import numpy as np
import onnxruntime as ort


MODEL_PATH = "osnet_x1_0.onnx"


print("Loading ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("✅ ONNX model loaded.")

print("Input:", session.get_inputs()[0].name)
print("Output:", session.get_outputs()[0].name)


# Same input shape used during ONNX export
dummy_input = np.random.randn(
    1, 3, 256, 128
).astype(np.float32)


print("Running inference...")

output = session.run(
    None,
    {
        session.get_inputs()[0].name: dummy_input
    }
)[0]


print("✅ Inference successful.")
print("Output shape:", output.shape)
print("Output dtype:", output.dtype)
print("First 5 values:", output.flatten()[:5])