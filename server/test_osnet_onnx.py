import torch
import torchreid


print("Loading OSNet...")

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

model.eval()

print("OSNet loaded successfully.")


# Dummy input matching your current preprocessing
dummy_input = torch.randn(1, 3, 256, 128)

print("Exporting OSNet to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    "osnet_x1_0.onnx",
    input_names=["input"],
    output_names=["features"],
    opset_version=17,
    dynamo=False
)

print("✅ ONNX export completed.")
print("Created: osnet_x1_0.onnx")