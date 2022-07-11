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

    # resized_imgs = [T.Resize(size=size)(orig_img) for size in (30, 50, 100, orig_img.size)]

    # center_crops = [T.CenterCrop(size=size)(orig_img) for size in (30, 50, 100, 300, orig_img.size)]

    (top_left, top_right, bottom_left, bottom_right, center) = T.FiveCrop(size=(300, 300))(orig_img)
    plot([top_left, top_right, bottom_left, bottom_right, center])