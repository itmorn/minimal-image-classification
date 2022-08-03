本文以pytorch提供的官方ResNet18为例进行梳理。

## 1.网络结构总览
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
## 2.图像预处理
![img.png](img.png)
### 2.1.训练阶段的预处理：
```python
dataset = torchvision.datasets.ImageFolder(
    traindir,
    presets.ClassificationPresetTrain(
        crop_size=train_crop_size,
        interpolation=interpolation,
        auto_augment_policy=auto_augment_policy,
        random_erase_prob=random_erase_prob,
    ),
)
```
```python
class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)
```
先使用RandomResizedCrop在原图一个随机位置截取块，然后缩放到224 * 244。
![img_1.png](img_1.png)
详情可查看https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py

然后可以根据数据增强的配置，进行其他的操作，比如水平翻转，融合之类的，但是官方的训练代码中都没有做。
最后使用Normalize对像素进行规范化处理

### 2.2.测试阶段的预处理：
```python
preprocessing = presets.ClassificationPresetEval(
    crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
)

dataset_test = torchvision.datasets.ImageFolder(
    valdir,
    preprocessing,
)
```
```python
class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)
```
然后首先将图像Resize到256 * 256，然后CenterCrop成224 * 224，最后Normalize

