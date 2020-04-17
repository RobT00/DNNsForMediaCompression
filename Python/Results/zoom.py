#!/usr/bin/env python3
from scipy.ndimage import rotate, zoom
from matplotlib import pyplot as plt
import cv2
import glob
import numpy as np
import os
import argparse
import results_util as util


def show_im(img: np.ndarray, grid: str = None):
    """
    Helper function to show image in matplotlib and overlay grid lines
    :param img: Image to show
    :param grid: Level of grid to overlay
    :return:
    """
    plt.figure()
    if grid == "major":
        plt.imshow(img, vmin=-1, vmax=1)
        plt.grid(which="both", linewidth=0.72, color="k")
    elif grid == "minor":
        plt.minorticks_on()
        plt.imshow(img, vmin=-1, vmax=1)
        plt.grid(which="both", linewidth=0.72, color="k")
        plt.tick_params(which="minor", length=0)
    else:
        plt.imshow(img)

    plt.show()


def run(base_path: str, images: list, single: bool = False, save: bool = False):
    """
    Main function for handling and coordinating processing of images
    :param base_path: Base directory for finding images
    :param images: Names of images to find / process
    :param single: If a single image is being processed
    :param save: If processed images are being saved, no plotting
    :return:
    """
    os.chdir(base_path)
    os.makedirs("output", exist_ok=True)
    output_dir = os.path.join(base_path, "output")
    files = os.listdir(base_path)
    q, im_name = base_path.split(os.sep)[-2:]
    if im_name == "kodim03":
        # For kodim03
        i_x1, i_y1 = (140, 140)
        i_x2, i_y2 = (260, 240)
    elif im_name == "kodim10":
        # For kodim10
        i_x1, i_y1 = (100, 300)
        i_x2, i_y2 = (200, 400)
    elif im_name == "kodim18":
        # For kodim10
        i_x1, i_y1 = (320, 60)
        i_x2, i_y2 = (380, 120)
    else:
        i_x1, i_y1 = (100, 100)
        i_x2, i_y2 = (300, 300)
    for image in images:
        # Load in the image
        image_path = [f for f in files if image in f][
            0
        ]  # Will throw exception if nothing found
        print(image_path)
        img = cv2.imread(image_path).astype("float32")
        if any(im_name == i for i in ["kodim10", "kodim18"]):
            img = np.rot90(img, 3)
        img /= 255.0

        if single:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Plot the original
            show_im(rgb_img)

            # Plot the original with major grid
            show_im(rgb_img, grid="major")

            # Plot the original with minor grid
            show_im(rgb_img, grid="minor")
        else:
            # Select the are of interest (using the above grids)
            # https://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
            im_c = img[i_y1:i_y2, i_x1:i_x2]
            # Plot selection
            if not save:
                rgb_img = cv2.cvtColor(im_c, cv2.COLOR_BGR2RGB)
                show_im(rgb_img)

            # Scale up the selection by factor scale_p
            # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
            scale_p = 300 if im_name == "kodim18" else 150
            width = int(im_c.shape[1] * scale_p / 100)
            height = int(im_c.shape[0] * scale_p / 100)
            dim = (width, height)

            im_c = cv2.resize(im_c, dim, interpolation=cv2.INTER_AREA)

            # Set offset for applying to original image
            # y_offset = 5
            # x_offset = 5

            if im_name == "kodim03":
                # For kodim03
                y_offset = 10
                x_offset = 10
                y1, y2 = (
                    img.shape[0] - y_offset - im_c.shape[0],
                    img.shape[0] - y_offset,
                )
                x1, x2 = x_offset, x_offset + im_c.shape[1]
            elif im_name == "kodim18":
                y_offset = 50
                x_offset = 10
                y1, y2 = (
                    img.shape[0] - y_offset - im_c.shape[0],
                    img.shape[0] - y_offset,
                )
                x1, x2 = x_offset, x_offset + im_c.shape[1]
            else:
                # For kodim10
                y_offset = 5
                x_offset = 5
                y1, y2 = y_offset, y_offset + im_c.shape[0]
                x1, x2 = x_offset, x_offset + im_c.shape[1]

            o_im = img.copy()

            # If alpha channels were to matter...

            # alpha_s = im_c[:, :, 2] / 255.0
            # alpha_l = 1.0 - alpha_s
            #
            #
            # for c in range(0, 3):
            #     o_im[y1:y2, x1:x2, c] = (alpha_s * im_c[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])

            # Draw box(es)
            # (0, 0, 205) -> medium blue
            # (255, 0, 0) -> red
            # See: https://www.rapidtables.com/web/color/index.html
            # Reverse because OpenCV is BGR
            colour = (0, 0, 200)
            if save:
                o_im *= 255.0
                im_c *= 255.0
            # https://stackoverflow.com/questions/23720875/how-to-draw-a-rectangle-around-a-region-of-interest-in-python
            cv2.rectangle(o_im, (i_x1, i_y1), (i_x2, i_y2), colour, thickness=2)
            cv2.rectangle(o_im, (x1, y1), (x2, y2), colour, thickness=2)

            # Overlay the selection
            o_im[y1:y2, x1:x2] = im_c

            if save:
                o_im = o_im.astype("uint8")
                save_path = os.path.join(output_dir, os.path.basename(image_path))
                cv2.imwrite(save_path, o_im)
            else:
                # Show overlaid images
                rgb_img = cv2.cvtColor(o_im, cv2.COLOR_BGR2RGB)
                show_im(rgb_img)
    if save:
        combined_filename = f"combined_{q}_{im_name}.png"
        # Create image with 3 images side by side
        try:
            os.remove(f"{output_dir}/{combined_filename}")
        except OSError:
            pass
        if im_name == "kodim03":
            corner = "top_left"
            horizontal = False
        else:
            corner = "bottom_left"
            horizontal = True
        # Ensure ordering: original, compressed, trained
        ims = list()
        for i in images:
            ims.append(glob.glob(f"{output_dir}/{i}*")[0])
        cat_im = util.concat_n_images(
            ims, horizontal=horizontal, img_corner=corner, gap=10
        )
        save_path = os.path.join(output_dir, combined_filename)
        cv2.imwrite(save_path, cat_im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--original",
        dest="original",
        action="store_true",
        help="Plotting only original, will show grids, use for selection",
        default=False,
    )
    # Specify the base image path
    # Image path expected to contain: original, compressed and trained
    parser.add_argument(
        "-p", "--image-path", dest="base_image_path", default=None, type=str
    )
    parser.add_argument(
        "-s",
        "--save",
        dest="save",
        help="Save the overlaid images",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    if args.original:
        images = ["original"]
    else:
        images = ["original", "compressed", "trained"]

    run(args.base_image_path, images, single=args.original, save=args.save)
