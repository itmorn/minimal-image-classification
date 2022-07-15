import torchvision.models as models
import torch

import onnx
import onnx.utils
import onnx.version_converter


# 定义数据+网络
data = torch.randn(2, 3, 256, 256)
net = models.resnet50()

# 导出
torch.onnx.export(
    net,
    data,
    'model.onnx',
    export_params=True,
    opset_version=8,
)

# 增加维度信息
model_file = 'model.onnx'
onnx_model = onnx.load(model_file)
onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)