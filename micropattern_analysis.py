import imageio.v3 as iio
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_triangle
from skimage.measure import regionprops
import skimage.morphology as morph
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from argparse import ArgumentParser
from time import time


def gather_files(path):
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                yield os.path.join(root, file)
    else:
        print("[-] No such file or directory")


def maximise_img_channels(img):
    return np.sum(img, axis=0)


def create_img_mask(img, dapi_img):
    if len(img.shape) == 2:
        print("[-] Selected single channel image")
        return
    dapi_img_mask = dapi_img > threshold_triangle(dapi_img)
    seed = np.copy(dapi_img_mask)
    seed[1:-1, 1:-1] = dapi_img_mask.max()
    dapi_img_mask = morph.reconstruction(seed, dapi_img_mask, method="erosion")
    return dapi_img_mask


def apply_img_mask(img, img_mask):
    img_channels = []
    for channel in img:
        img_channels.append(channel * img_mask)
    img_channels = np.array(img_channels)
    return img_channels


def get_center_of_mass(img_mask, dapi_img):
    properties = regionprops(img_mask.astype(int), dapi_img)
    center_of_mass = np.array(properties[0].centroid).astype(int)
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


def get_channel_values_of_cords(img, dapi_channel_number):
    return np.transpose(img[0:dapi_channel_number].reshape((dapi_channel_number, img.shape[-1] * img.shape[-2])))


def group_distances(df, channel_names):
    distance_intensities = df[df[channel_names[0]] > 0].groupby("Distances", sort=True).mean()
    distance_intensities.reset_index(inplace=True)
    return distance_intensities


def scale_distances(df, mean_res):
    df["Distances"] = df["Distances"] * mean_res
    return df


def smooth_distances(df, channel_names):
    for channel in channel_names:
        df[channel] = gaussian_filter(df[channel], 4)
    return df


def plot_data(img, df, channel_names, file_path, mean_res, save=False):
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * mean_res))

    (fig, ax) = plt.subplots(len(channel_names), 2, figsize=(10, 15))
    for i in range(len(ax)):
        ax[i][0].imshow(img[i], cmap="twilight")
        ax[i][0].set_ylabel(channel_names[i])
        ax[i][0].xaxis.set_major_formatter(ticks)
        ax[i][0].yaxis.set_major_formatter(ticks)
        sns.lineplot(df, x="Distances", y=channel_names[i], hue="Culture Condition", ax=ax[i][1])
    fig.supxlabel("Distance [µm]")
    if save:
        plt.savefig(f"{file_path}_plot.png")
    else:
        plt.show()
        plt.close("all")


def main():
    parser = ArgumentParser()
    parser.add_argument("pathfile", help="specify file or filepath...")
    parser.add_argument("-d", "--dapi",
                        help="specify channel number of DAPI channel, if left empty assumes position of last channel",
                        type=int)
    parser.add_argument("-v", "--verbose", help="output verbosity", action="store_true")
    parser.add_argument("--debug", help="for debug purpose", action="store_true")
    parser.add_argument("-s", "--save-data", help="store data in csv", action="store_true")
    parser.add_argument("-p", "--save-plot", help="save plot as a png", action="store_true")
    parser.add_argument("-c", "--channel-names", help="provide list of channel names",
                        nargs="+")
    # add argument to specify which images should be plotted
    # add argument to save plots as pdf

    args = parser.parse_args(sys.argv[1:])

    frames = []
    images = []
    file_paths = []
    time_points = []
    t0 = time()
    time_points.append(t0)

    for file in gather_files(args.pathfile):
        try:
            img = iio.imread(file)
            images.append(img)
            file_paths.append(file)
            if len(img.shape) == 4:
                img = maximise_img_channels(img)

            if args.dapi:
                dapi_channel_number = args.dapi - 1
            else:
                dapi_channel_number = img.shape[0] - 1

            dapi_img = img[dapi_channel_number]
            img_mask = create_img_mask(img, dapi_img)
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
                mean_res = 0.69

            df_mini = group_distances(df_mini, channel_names)
            df_mini = scale_distances(df_mini, mean_res)
            df_mini = smooth_distances(df_mini, channel_names)
            df_mini["Culture Condition"] = np.repeat(os.path.basename(os.path.dirname(file)),
                                                     df_mini.shape[0])

            plot_data(img, df_mini, channel_names[:-1], file, mean_res, save=args.save_plot)

            frames.append(df_mini)

            time_points.append(time() - time_points[-1])
            print(f"[+] Finished image in {round(time_points[-1])} s") if args.verbose else ""
        except OSError:
            print("[-] Folder contained non-readable image/file")
    if args.debug:
        df = pd.concat(frames)
        plot_data(images[0], df,
                  ["PAX6", "SOX10", "ISL12"],
                  "merge", 0.6, save=args.save_plot)
    time_points.append(time() - t0)
    print(f"[+] Finished process in {round(time_points[-1])} s") if args.verbose else ""


if __name__ == '__main__':
    main()
