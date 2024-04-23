import imageio.v3 as iio
import numpy as np
from scipy.ndimage import gaussian_filter
import skimage.filters as filters
from skimage.measure import regionprops
import skimage.morphology as morph
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from argparse import ArgumentParser
from time import time
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
width, height = plt.rcParams.get('figure.figsize')


def gather_files(paths):
    """
    Implicit function: returns all files of given path in one dimension (flat hierarchy)

    :param paths:
    :return:
    """
    for pa in paths:
        if os.path.isfile(pa):
            yield pa
        elif os.path.isdir(pa):
            for root, _, files in os.walk(pa):
                for file in files:
                    yield os.path.join(root, file)
        else:
            print("[-] No such file or directory")


def get_files_and_images(paths):
    """
    Implicit function: returns dictionary of file names as keys and images (numpy matrix) of given path

    :param paths:
    :return:
    """
    res_dir = {}
    for file in gather_files(paths):
        img = iio.imread(file)
        res_dir.update({file: img})
    return res_dir


def maximise_img_channels(img):
    """
    Explicit function: returns image with "brightest"/maximum pixel values
    Implicit use: effectively can transform z-stack image into single stack

    :param img:
    :return:
    """
    return np.max(img, axis=0)


def normalize_image(img, bits=8):
    """
    Explicit function: normalizes pixel values (ranging from 0 to 1)

    :param img:
    :param bits:
    :return:
    """
    bits = (2 ** bits) - 1
    return img / bits


def maximise_and_normalize(img, bits=8):
    """
    Explicit function: uses first maximise_img_channels() and then normalizes_image()

    :param img:
    :param bits:
    :return:
    """
    return normalize_image(maximise_img_channels(img), bits)


def create_solid_img_mask(img, dapi_img, threshold_fun):
    """
    Implicit function: Creates a solid image mask (without holes and removes small objects) from
    image with given function

    :param img:
    :param dapi_img:
    :param threshold_fun:
    :return:
    """
    if len(img.shape) == 2:
        print("[-] Selected single channel image")
        return
    dapi_img_mask = dapi_img > threshold_fun(dapi_img)
    seed = np.copy(dapi_img_mask)
    seed[1:-1, 1:-1] = dapi_img_mask.max()
    dapi_img_mask = morph.remove_small_objects(dapi_img_mask, 200)
    dapi_img_mask = morph.reconstruction(seed, dapi_img_mask, method="erosion")
    return dapi_img_mask


def create_img_mask_multiotsu(
        img,
        dapi_img_num,
        num_classes=4,
        threshold_index=1
):
    """
    Explicit function: returns a binary image mask of the given DAPI channel by using Multi-Otsu
    algorithm

    :param img:
    :param dapi_img_num:
    :param num_classes:
    :param threshold_index:
    :return:
    """
    thresholds = filters.threshold_multiotsu(img[dapi_img_num], classes=num_classes)
    dapi_threshold = thresholds[threshold_index]
    return img[dapi_img_num] > dapi_threshold


def apply_multiotsu_to_image(
        img,
        dapi_img_num,
        num_classes=4,
        threshold_index=1
):
    """
    Implicit function: uses create_img_mask_multiotsu() to apply binary mask to image

    :param img:
    :param dapi_img_num:
    :param num_classes:
    :param threshold_index:
    :return:
    """
    img = apply_img_mask(
        img,
        create_img_mask_multiotsu(
            img,
            dapi_img_num,
            num_classes,
            threshold_index
        )
    )
    return img


def apply_multiotsu_to_channel(
        img,
        channel_num,
        num_classes=4,
        threshold_index=1
):
    """
    Implicit function: uses create_img_mask_multiotsu() to apply binary mask to given channel in image

    :param img:
    :param channel_num:
    :param num_classes:
    :param threshold_index:
    :return:
    """
    img[channel_num] = apply_img_mask(
        img[channel_num],
        create_img_mask_multiotsu(
            img,
            channel_num,
            num_classes,
            threshold_index
        )
    )
    return img


def apply_img_mask(img, img_mask):
    """
    Explicit function: returns image with applied binary mask

    :param img:
    :param img_mask:
    :return:
    """
    return img * img_mask


def get_region_prop_from_channel(img_channel):
    """
    Explicit function: return region prop (single) from single (binary) labeled channel/image

    :param img_channel:
    :return:
    """
    return regionprops(np.array(img_channel > 0).astype(int))[0]


def get_center_of_mass(prop):
    """
    Explicit function: return center of mass (centroid) from associated region prop

    :param prop:
    :return:
    """
    center_of_mass = np.array(prop.centroid).astype(int)
    return center_of_mass


def expand_coordinate_matrix(dapi_img):
    cords = np.mgrid[0:dapi_img.shape[0], 0:dapi_img.shape[1]]  # All points in a 3D grid within the given ranges
    cords = np.rollaxis(cords, 0, 3)  # Make the 0th axis into the last axis
    cords = cords.reshape(
        (dapi_img.shape[0] * dapi_img.shape[1], 2))  # Now you can safely reshape while preserving order
    return cords


def calculate_distances(cords, center_of_mass):
    dist = np.linalg.norm(cords - center_of_mass, axis=1).astype(int)
    return dist


def get_distances(image_channel):
    """
    Implicit function: uses calculate_distances() with implicit inputs

    :param image_channel:
    :return:
    """
    cords = expand_coordinate_matrix(image_channel)
    centroid = get_center_of_mass(get_region_prop_from_channel(image_channel))
    return calculate_distances(cords, centroid)


def generate_data_frame(cords, img, channel_names, **kwargs):
    channel_values_of_cords = get_channel_values_of_cords(img, len(channel_names))
    cords_grey_dict = {
        "x": cords[:, 0],
        "y": cords[:, 1],
    }
    for i in range(channel_values_of_cords.shape[1]):
        cords_grey_dict.update({channel_names[i]: channel_values_of_cords[:, i]})
    for k in kwargs.keys():
        cords_grey_dict.update({k: kwargs.get(k)})
    df = pd.DataFrame(cords_grey_dict)
    return df


def image_mean_res(file):
    img_props = iio.improps(file)
    img_meta = iio.immeta(file)
    mean_res = 1
    if img_props.spacing:
        mean_res = np.mean(1 / np.array(img_props.spacing))
    elif img_meta["ScanInformation"]["SampleSpacing"]:
        mean_res = img_meta["ScanInformation"]["SampleSpacing"]
    return mean_res


def get_channel_values_of_cords(img, dapi_channel_number):
    return np.transpose(img[0:dapi_channel_number].reshape((dapi_channel_number, img.shape[-1] * img.shape[-2])))


def average_distances(df, groups=None):
    if groups is None:
        groups = ["Distances"]
    distance_intensities = df.groupby(groups, sort=True).mean()
    distance_intensities.reset_index(inplace=True)
    return distance_intensities


def scale_distances(df, mean_res=None, file=None):
    if not mean_res and file:
        mean_res = image_mean_res(file)
    df["Distances"] = df["Distances"] * mean_res
    return df


def smooth_distances(df, channel_names, sigma=4):
    for channel in channel_names:
        df[channel] = gaussian_filter(df[channel], sigma)
    return df


def plot_data(img, df, channel_names, file_path, mean_res, save=False):
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(x * mean_res)))

    (fig, ax) = plt.subplots(len(channel_names), 2, figsize=(10, 15))
    for i in range(len(ax)):
        im = ax[i][0].imshow(img[i], cmap="twilight")
        ax[i][0].set_ylabel(channel_names[i])
        ax[i][0].xaxis.set_major_formatter(ticks)
        ax[i][0].yaxis.set_major_formatter(ticks)
        pos = fig.add_axes([0.93, 0.1, 0.02, 0.35])
        fig.colorbar(im, cax=pos)
        sns.lineplot(df, x="Distances", y=channel_names[i], ax=ax[i][1])
        ax[i][1].set_ylabel("")
        ax[i][1].set_xlabel("")
    fig.supxlabel("Distance [µm]")
    if save:
        plt.savefig(f"{file_path}_plot.png")
    else:
        plt.show()
        plt.close("all")


def plot_merge_data(images, df, channel_names, file_path, mean_res, save=False):
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(round(x * mean_res)))

    (fig, axes) = plt.subplots(len(channel_names), len(images) + 1, figsize=(20, 15))
    for i in range(len(axes)):
        for ax, key in zip(axes[i], images.keys()):
            img = images.get(key)[i]
            im = ax.imshow(img, cmap="twilight")
            ax.set_ylabel(channel_names[i])
            ax.xaxis.set_major_formatter(ticks)
            ax.yaxis.set_major_formatter(ticks)
            ax.set_title(key)
            pos = fig.add_axes([0.93, 0.1, 0.02, 0.35])
            fig.colorbar(im, cax=pos)
        sns.lineplot(df, x="Distances", y=channel_names[i], hue="Culture Condition", ax=axes[i][-1])
        axes[i][-1].set_ylabel("")
        axes[i][-1].set_xlabel("")
    fig.supxlabel("Distance [µm]")
    if save:
        plt.savefig(f"{file_path}_plot.png")
    else:
        plt.show()
        plt.close("all")


# TODO: remove or improve function
def main():
    parser = ArgumentParser()
    parser.add_argument("pathfile", help="specify file or filepath...")
    parser.add_argument("-d", "--dapi",
                        help="specify channel number of DAPI channel, if left empty assumes position of last channel",
                        type=int)
    parser.add_argument("-v", "--verbose", help="output verbosity", action="store_true")
    parser.add_argument("--debug", help="for debug purpose", action="store_true")
    parser.add_argument("-s", "--save-data", help="store data in csv", action="store_true")
    parser.add_argument("-p", "--save-plot",
                        help="save plot as a png, after program is executed it is presented which images"
                             " should be plotted",
                        action="store_true")
    parser.add_argument("-c", "--channel-names", help="provide list of channel names",
                        nargs="+")
    parser.add_argument("--normalize", help="normalizes data", action="store_true")
    parser.add_argument("--smoothing-sigma", help="specify gaussian sigma for smoothing", type=float)
    # add argument to specify which images should be plotted
    # add argument to save plots as pdf

    args = parser.parse_args(sys.argv[1:])

    frames = []
    images = {}
    file_paths = []
    time_points = []
    t0 = time()
    time_points.append(t0)

    plot_files = []

    if args.save_plot:
        print()
        print("Specify which files should be included to be saved as plots:")
        print()
        files = [file for file in gather_files(args.pathfile)]
        for i in range(len(files)):
            print(f"({i + 1})\t\t{files[i]}")
        plots = input("Select files to be saved as plots (separate with empty space ' '): ")
        for i in map(int, plots.split(" ")):
            plot_files.append(files[i - 1])

    for file in gather_files(args.pathfile):
        try:
            img = iio.imread(file)

            if len(img.shape) == 4:
                img = maximise_img_channels(img)

            if len(file_paths) == 0:
                images.update({os.path.basename(os.path.dirname(file)): img})
            elif os.path.dirname(file) != os.path.dirname(file_paths[-1]):
                images.update({os.path.basename(os.path.dirname(file)): img})

            file_paths.append(file)

            if args.dapi:
                dapi_channel_number = args.dapi - 1
            else:
                dapi_channel_number = img.shape[0] - 1

            dapi_img = img[dapi_channel_number]
            img_mask = create_solid_img_mask(img, dapi_img, filters.threshold_triangle)
            applied_img_mask = apply_img_mask(img, img_mask)
            center_of_mass = get_center_of_mass(img_mask, dapi_img)
            cords = expand_coordinate_matrix(dapi_img)

            dist = calculate_distances(cords, center_of_mass)

            if args.channel_names:
                channel_names = args.channel_names
                if len(channel_names) != dapi_channel_number + 1:
                    channel_names.append("DAPI")
            else:
                channel_names = [f"Channel {i + 1}" for i in range(dapi_channel_number + 1)]

            df_mini = generate_data_frame(cords, applied_img_mask, channel_names, Distances=dist)

            csv_file_name = f"{file}.csv"
            if not os.path.isfile(csv_file_name) and args.save_data:
                df_mini.to_csv(csv_file_name, index=False)

            img_props = iio.improps(file)
            img_meta = iio.immeta(file)
            if img_props.spacing:
                mean_res = np.mean(1 / np.array(img_props.spacing))
            elif img_meta["ScanInformation"]["SampleSpacing"]:
                mean_res = img_meta["ScanInformation"]["SampleSpacing"]
            else:
                mean_res = 1

            df_mini = group_distances(df_mini, channel_names)
            df_mini = scale_distances(df_mini, mean_res)
            if args.smoothing_sigma:
                df_mini = smooth_distances(df_mini, channel_names, sigma=args.smoothing_sigma)
            else:
                df_mini = smooth_distances(df_mini, channel_names)
            df_mini["Culture Condition"] = np.repeat(os.path.basename(os.path.dirname(file)),
                                                     df_mini.shape[0])

            if args.save_plot:
                if file in plot_files:
                    plot_data(img, df_mini, channel_names[:-1], file, mean_res, save=args.save_plot)
            else:
                plot_data(img, df_mini, channel_names[:-1], file, mean_res)

            frames.append(df_mini)

            time_points.append(time() - time_points[-1])
            print(f"[+] Finished image in {round(time_points[-1])} s") if args.verbose else ""
        except OSError:
            print("[-] Folder contained non-readable image/file")
    if args.debug:
        df = pd.concat(frames)
        plot_merge_data(images, df,
                        args.channel_names[:-1],
                        "merged", 0.69, save=args.save_plot)
    time_points.append(time() - t0)
    print(f"[+] Finished process in {round(time_points[-1])} s") if args.verbose else ""


if __name__ == '__main__':
    main()
