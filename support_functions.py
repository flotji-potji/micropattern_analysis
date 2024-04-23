import matplotlib.pyplot as plt
import math

from micropattern_analysis import *


def print_image_dim_explanation():
    print("(3, 4, 1024, 1024)\n"
          " |  |    |     |  \n"
          " |  |    +-----+-- X & Y 'coordinates' of image\n"
          " |  +------------- channels of image\n"
          " +---------------- z-stacks of image\n")


def plot_multi_otsu_thresholds(image, channel, classes):
    square = classes // 2 if classes > 3 else classes - 1

    fig, axes = plt.subplots(square, square)
    for i, ax in enumerate(axes.flat):
        if i < (classes - 1):
            ax.imshow(create_img_mask_multiotsu(
                image,
                channel,
                num_classes=classes,
                threshold_index=i), cmap="grey"
            )
            ax.set_title(f"threshold index = {i}")
        ax.axis('off')
    fig.suptitle(f"Multi-Otsu Thresholding with {classes} classes")
    plt.show()


def plot_full_image(axes, img):
    for stack in img:
        plot_all_channels(axes, stack)


def plot_all_channels(axes, img):
    for channel, ax in zip(img, axes.flat):
        ax.imshow(channel, cmap="grey")
        ax.axis("off")


def plot_images(images, channel=0, fig_size=None):
    if fig_size is None:
        fig_size = (10, 10)
    square = math.ceil(math.sqrt(len(images)))
    fig, axes = plt.subplots(square, square, figsize=fig_size)
    for i, (ax, img) in enumerate(zip(axes.flat, images)):
        ax.imshow(img[channel], cmap="grey")
        ax.set_title(f'Index: {i}')
        ax.axis("off")
    return fig, axes
