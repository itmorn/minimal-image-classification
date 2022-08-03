本文以pytorch提供的官方ResNet18为例进行梳理。

# 1.网络结构总览
首先观察一下其网络结构：
![](resnet.onnx.png)

ResNet包含四个核心模块（如上图的四个框），其完整的前向传播过程为：
```python
x = self.conv1(x)
x = self.bn1(x)
x = self.relu(x)
x = self.maxpool(x)

x = self.layer1(x)
x = self.layer2(x)
x = self.layer3(x)
x = self.layer4(x)

x = self.avgpool(x)
x = torch.flatten(x, 1)
x = self.fc(x)
```
