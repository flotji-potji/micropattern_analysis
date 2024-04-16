import matplotlib.pyplot as plt
from micropattern_analysis import *


def print_image_dim_explanation():
    print("(3, 4, 1024, 1024)\n"
          " |  |    |     |  \n"
          " |  |    +-----+-- X & Y 'coordinates' of image\n"
          " |  +------------- channels of image\n"
          " +---------------- z-stacks of image\n")


def plot_multi_otsu_thresholds(image, classes):
    square = classes // 2 if classes > 3 else classes - 1

    fig, axes = plt.subplots(square, square)
    for i, ax in enumerate(axes.flat):
        if i < (classes - 1):
            ax.imshow(create_dapi_img_mask_multiotsu(image,
                                                     3,
                                                     num_classes=classes,
                                                     threshold_index=i), cmap="grey")
            ax.set_title(f"threshold index = {i}")
        ax.axis('off')
    fig.suptitle(f"Multi-Otsu Thresholding with {classes} classes")
    plt.show()


def plot_full_image(img):
    # TODO: fix global variable problem for module
    global axes
    for stack in img:
        plot_all_channels(stack)


def plot_all_channels(img):
    global axes
    for channel, ax in zip(img, axes.flat):
        ax.imshow(channel, cmap="grey")
        ax.axis("off")
