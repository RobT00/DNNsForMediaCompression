"""
File used for generating images of various qualities using a specified codec
"""
import os
from PIL import Image
import argparse
import glob
from timeit import default_timer as timer
from datetime import timedelta


def compress(
    in_dir: str,
    out_dir: str,
    quality: str,
    img_format: str = "png",
    compression: str = "JPEG",
    save_extension: str = "jpg",
) -> timedelta:
    """
    Function used to compress and save images under the specified compression scheme
    :param in_dir: Path of input images
    :param out_dir: Path to save compressed images in
    :param quality: List of qualities to use
    :param img_format: File extension of input images
    :param compression: Compression scheme to apply
    :param save_extension: Container to save compressed images in
    :return: Average time to generate each compressed image
    """
    total_images = 0
    output_time = 0.0
    quality_list = list()
    if type(quality) is str:
        for i in quality.split(","):
            quality_list.append(int(i))
    qualities = len(quality_list)
    for q in quality_list:
        for filename in glob.glob(in_dir + f"/*.{img_format}"):
            start = timer()
            img = Image.open(filename)
            img_comp = os.path.join(
                out_dir,
                f"{os.path.basename(filename.split('.')[0])}_{100 - q}.{save_extension}",
            )
            img.save(img_comp, format=compression, quality=q)
            end = timer()
            total_images += 1
            output_time += (end - start) * 1000

    print(
        f"Successfully generated {qualities} quality levels for "
        f"{total_images // qualities} images: {total_images} total"
    )
    return timedelta(milliseconds=output_time / total_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--in-dir",
        dest="in_dir",
        help="Path to images to be compressed",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        help="Path to save compressed images",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-q",
        "--quality",
        dest="quality",
        help="Quality values to save images at - comma separated list",
        default="5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 23, 25, 30",
        type=str,
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    avg_time = compress(args.in_dir, args.out_dir, args.quality)
    print(f"Avg time: {avg_time}")
