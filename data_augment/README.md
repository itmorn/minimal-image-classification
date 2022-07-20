# 按照固定方式转换图像
## Pad
fills image borders with some pixel values.

![img_Pad.png](imgs/img_Pad.png)

## Resize
将图像Resize成指定尺寸，还可以选择h和w，以及插值方式

![img_Resize.png](imgs/img_Resize.png)

## CenterCrop
crops the given image at the center.可以指定h和w

![img_CenterCrop.png](imgs/img_CenterCrop.png)

## FiveCrop
crops the given image into four corners and the central crop.返回5张图（四个角+中央）

![img_FiveCrop.png](imgs/img_FiveCrop.png)

## Grayscale
converts an image to grayscale

![img_Grayscale.png](imgs/img_Grayscale.png)


# 按照随机方式转换图像
## ColorJitter
randomly changes the brightness, saturation, and other properties of an image. 亮度、对比度、饱和度、色调

![img_ColorJitter.png](imgs/img_ColorJitter.png)

## GaussianBlur
randomly performs gaussian blur transform on an image. 高斯模糊

![img_GaussianBlur.png](imgs/img_GaussianBlur.png)

## RandomPerspective
randomly performs random perspective transform on an image.透视变换。可以指定变换尺度、执行概率

![img_RandomPerspective.png](imgs/img_RandomPerspective.png)

## RandomRotation
randomly rotates an image with random angle.旋转

![img_RandomRotation.png](imgs/img_RandomRotation.png)

## RandomAffine
randomly performs random affine transform on an image.仿射变换

![img_RandomAffine.png](imgs/img_RandomAffine.png)

## RandomCrop
randomly crops an image at a random location.随机裁剪

![img_RandomCrop.png](imgs/img_RandomCrop.png)

## RandomResizedCrop
randomly crops an image at a random location, and then resizes the crop to a given size.
随机截取一个区域，然后resize到指定尺寸

![img_RandomResizedCrop.png](imgs/img_RandomResizedCrop.png)

## RandomInvert
randomly inverts the colors of the given image.图像反相

![img_RandomInvert.png](imgs/img_RandomInvert.png)

## RandomPosterize
randomly posterizes the image by reducing the number of bits of each color channel.
减少颜色通道的bit位数，每个颜色通道上，取值为【0~255】为8个bit；当bit位数减少时，就相当于对色彩做了离散

![img_RandomPosterize.png](imgs/img_RandomPosterize.png)

## RandomSolarize
randomly solarizes the image by inverting all pixel values above the threshold.
图像日光化，大于等于设定阈值的像素会被取反相，阈值设置为0时等价于RandomInvert

![img_RandomSolarize.png](imgs/img_RandomSolarize.png)

## RandomAdjustSharpness
randomly adjusts the sharpness of the given image.
锐化

![img_RandomAdjustSharpness.png](imgs/img_RandomAdjustSharpness.png)

## RandomAutocontrast
randomly applies autocontrast to the given image.
自动对比度

![img_RandomAutocontrast.png](imgs/img_RandomAutocontrast.png)

## RandomEqualize
randomly equalizes the histogram of the given image.
直方图均衡化

![img_RandomEqualize.png](imgs/img_RandomEqualize.png)

## AutoAugment
transform automatically augments data based on a given auto-augmentation policy.
采用已有数据的数据增广方案

![img_AutoAugment.png](imgs/img_AutoAugment.png)

## RandAugment
transform automatically augments the data.
随机增广，可以控制变化的幅度和种类数

![img_RandAugment.png](imgs/img_RandAugment.png)

## TrivialAugmentWide
https://arxiv.org/pdf/2103.10158.pdf 经验总结的一套增广方案

![img_TrivialAugmentWide.png](imgs/img_TrivialAugmentWide.png)

## AugMix
https://arxiv.org/pdf/1912.02781.pdf 常用的图像融合，提升模型鲁棒性的增广方案

![img_AugMix.png](imgs/img_AugMix.png)

## RandomHorizontalFlip
performs horizontal flip of an image, with a given probability. 水平翻转

![img_RandomHorizontalFlip.png](imgs/img_RandomHorizontalFlip.png)

## RandomVerticalFlip
performs vertical flip of an image, with a given probability. 竖直翻转

![img_RandomVerticalFlip.png](imgs/img_RandomVerticalFlip.png)

## RandomApply
randomly applies a list of transforms, with a given probability.
按照一定概率，执行一系列的转换

![img_RandomApply.png](imgs/img_RandomApply.png)

# 参考
https://pytorch.org/vision/stable/auto_examples/plot_transforms.html

