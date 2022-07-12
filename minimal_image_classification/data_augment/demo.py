"""
@Auth: itmorn
@Date: 2022/7/11-15:33
@Email: 12567148@qq.com
"""
# sphinx_gallery_thumbnail_path = "../../gallery/assets/transforms_thumbnail.png"

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T

plt.rcParams["savefig.bbox"] = 'tight'
orig_img = Image.open(Path('assets') / 'astronaut.jpg')
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # padded_imgs = [T.Pad(padding=padding)(orig_img) for padding in (3, 10, 30, 50)]
    # plot(padded_imgs)

    # resized_imgs = [T.Resize(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]
    # plot(resized_imgs)

    # center_crops = [T.CenterCrop(size=size)(orig_img) for size in (30, 50, 100, 300, orig_img.size)]
    # plot(center_crops)

    # (top_left, top_right, bottom_left, bottom_right, center) = T.FiveCrop(size=(300, 300))(orig_img)
    # plot([top_left, top_right, bottom_left, bottom_right, center])

    # gray_img = T.Grayscale()(orig_img)
    # plot([gray_img], cmap='gray')

    # jitter = T.ColorJitter(brightness=.5, hue=.3)
    # jitted_imgs = [jitter(orig_img) for _ in range(4)]
    # plot(jitted_imgs)

    # blurrer = T.GaussianBlur(kernel_size=(15, 59), sigma=(5, 15))
    # blurred_imgs = [blurrer(orig_img) for _ in range(4)]
    # plot(blurred_imgs)

    # perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)
    # perspective_imgs = [perspective_transformer(orig_img) for _ in range(4)]
    # plot(perspective_imgs)

    # rotater = T.RandomRotation(degrees=(0, 360))
    # rotated_imgs = [rotater(orig_img) for _ in range(4)]
    # plot(rotated_imgs)

    # affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75),shear=50)
    # affine_imgs = [affine_transfomer(orig_img) for _ in range(4)]
    # plot(affine_imgs)

    # cropper = T.RandomCrop(size=(400, 400))
    # crops = [cropper(orig_img) for _ in range(4)]
    # plot(crops)

    # resize_cropper = T.RandomResizedCrop(size=(200, 300), scale=(0.5, 1.0))
    # resized_crops = [resize_cropper(orig_img) for _ in range(4)]
    # plot(resized_crops)

    # inverter = T.RandomInvert(p=0.5)
    # invertered_imgs = [inverter(orig_img) for _ in range(4)]
    # plot(invertered_imgs)

    # posterizer = T.RandomPosterize(bits=2)
    # posterized_imgs = [posterizer(orig_img) for _ in range(4)]
    # plot(posterized_imgs)

    # solarizer = T.RandomSolarize(threshold=192.0)
    # solarized_imgs = [solarizer(orig_img) for _ in range(4)]
    # plot(solarized_imgs)

    # sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=10, p=1)
    # sharpened_imgs = [sharpness_adjuster(orig_img) for _ in range(2)]
    # plot(sharpened_imgs)

    # autocontraster = T.RandomAutocontrast(p=1)
    # autocontrasted_imgs = [autocontraster(orig_img) for _ in range(1)]
    # plot(autocontrasted_imgs)

    # equalizer = T.RandomEqualize(p=1)
    # equalized_imgs = [equalizer(orig_img) for _ in range(1)]
    # plot(equalized_imgs)

    # policies = [T.AutoAugmentPolicy.CIFAR10, T.AutoAugmentPolicy.IMAGENET, T.AutoAugmentPolicy.SVHN]
    # augmenters = [T.AutoAugment(policy) for policy in policies]
    # imgs = [
    #     [augmenter(orig_img) for _ in range(4)]
    #     for augmenter in augmenters
    # ]
    # row_title = [str(policy).split('.')[-1] for policy in policies]
    # plot(imgs, row_title=row_title)

    # augmenter = T.RandAugment()
    # imgs = [augmenter(orig_img) for _ in range(4)]
    # plot(imgs)

    # augmenter = T.TrivialAugmentWide()
    # imgs = [augmenter(orig_img) for _ in range(4)]
    # plot(imgs)

    augmenter = T.AugMix(severity = 10, mixture_width = 10,chain_depth=3)
    imgs = [augmenter(orig_img) for _ in range(1)]
    plot(imgs)
