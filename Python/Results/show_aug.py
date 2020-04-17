"""
File to show augmentations used in training
Also used for generating images of individual frames from videos
Author: Robert Trew
"""
import os
import code_util
import results_util
import argparse
import cv2
import glob
import shutil


def main(filename: str, video: bool, output_dir: str, do_aug: bool = True):
    """
    Main function, to invoke sub-functions as required
    :param filename: File to apply augmentations to
    :param video: Specify if processing image or video
    :param output_dir: Directory to output
    :return:
    """
    colour_space = "BGR" if video else "RGB"
    data_man = code_util.DataManagement(
        None, video, None, None, output_dir, None, colour_space
    )  # Create Basic instance
    # Make temp dir
    os.makedirs(os.path.join(data_man.out_path, "temp"), exist_ok=False)
    if do_aug:
        augmentations = {
            "rotate": 40,
            "brightness": -0.1,
            "noise": "gaussian",
            "contrast": 1.25,
        }
    else:
        # Showing a single frame from a video
        augmentations = dict()
    if video:
        frame = [338]  # football_422  # [2] akiyo  # [122] foreman
        process_video(data_man, filename, frame, augmentations)
    else:
        process_image(data_man, filename, augmentations)
    combined_filename = "combined.png"
    # Create side by side image
    try:
        os.remove(f"{output_dir}/{combined_filename}")
    except OSError:
        pass
    ims = glob.glob(f"{output_dir}/temp/*.png")
    cat_im = results_util.concat_n_images(
        ims, horizontal=True, img_corner="bottom_left", gap=10, text=False
    )
    save_path = os.path.join(output_dir, combined_filename)
    cv2.imwrite(save_path, cat_im)
    # Remove temp folder
    shutil.rmtree(os.path.join(data_man.out_path, "temp"))


def process_video(helper: code_util, vid: str, frame: list, augmentations: dict):
    """
    Apply augmentations to specified frame in a video
    :param helper: Instance of DataManagement class
    :param vid: Path of video
    :param frame: Frame to show augmentations on
    :param augmentations: Augmentations to apply
    :return:
    """
    to_save = helper.deprocess_image(
        *helper.preprocess_video(helper.load_video(vid), get_frames=frame)
    )
    cv2.imwrite(os.path.join(helper.out_path, "temp", "1_original.png"), to_save)
    for i, (aug_name, aug) in enumerate(augmentations.items(), start=2):
        to_save = helper.deprocess_image(
            *helper.preprocess_video(
                helper.load_video(vid), get_frames=frame, mode={aug_name: aug}
            )
        )
        cv2.imwrite(
            os.path.join(helper.out_path, "temp", f"{i}_{aug_name}.png"), to_save
        )


def process_image(helper: code_util, img: str, augmentations: dict):
    """
    Apply augmentations to an image
    :param helper: Instance of DataManagement class
    :param img: Path of image
    :param augmentations: Augmentations to apply
    :return:
    """
    to_save = helper.deprocess_image(helper.preprocess_image(img))
    cv2.imwrite(os.path.join(helper.out_path, "temp", "original.png"), to_save)
    for aug_name, aug in augmentations.items():
        to_save = helper.deprocess_image(
            helper.preprocess_image(img, mode={aug_name: aug})
        )
        cv2.imwrite(os.path.join(helper.out_path, "temp", f"{aug_name}.png"), to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--out-dir",
        dest="output_dir",
        help="Directory to output to",
        default=None,
        type=str,
    )

    parser.add_argument(
        "-f", "--filename", dest="filename", help="File to use", default=None, type=str
    )
    parser.add_argument(
        "-i",
        "--image",
        dest="video",
        action="store_false",
        help="Generate metrics for images (default is video)",
        default=True,
    )

    args = parser.parse_args()
    # Create a dictionary from the supplier args namespace
    # args_dict = vars(args)

    main(args.filename, args.video, args.output_dir)
