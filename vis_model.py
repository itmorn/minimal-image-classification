"""
@Auth: itmorn
@Date: 2022/8/2-17:23
@Email: 12567148@qq.com
"""
import torch
import onnx.shape_inference
import onnx.version_converter
from torchvision.models import resnet18, ResNet18_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

data = torch.rand(2, 3, 224, 224)
torch.save(model, 'resnet.pth')
torch.onnx.export(model, data, 'resnet.onnx', opset_version=16, input_names=['input'], output_names=['output'])
print("done")

# 增加维度信息
model_file = 'resnet.onnx'
onnx_model = onnx.load(model_file)
onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)