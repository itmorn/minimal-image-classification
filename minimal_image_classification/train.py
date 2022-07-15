import cv2
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet50, ResNet50_Weights
import torch
import numpy as np

img = read_image("data_augment/assets/20715152805.png",mode = ImageReadMode.RGB)
img2 = cv2.imread("data_augment/assets/20715152805.png")
img2 = np.transpose(img2,[2,0,1])[::-1].copy()
img2 = torch.from_numpy(img2)
# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img2).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
torch.save(model,"a.pt")
print(f"{category_name}: {100 * score:.1f}%")

import onnx
x = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    x,
    "srcnn.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output'])

onnx_model = onnx.load("srcnn.onnx")
onnx.save(onnx.shape_inference.infer_shapes(onnx_model), "srcnn.onnx")