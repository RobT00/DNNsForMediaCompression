#!/usr/bin/env python3
"""
Main file for producing results in 5E1 project
Author: Robert Trew
"""
import os
import argparse
import image_results
import video_results


def main(script_dir: str, video: bool = True, **kwargs):
    if video:
        base_path = ""
        compressed_data_path = os.path.join(base_path, kwargs.get("codec", "H264"))
        original_data_path = ""

        video_results.vmaf_score(
            original_data_path, compressed_data_path, script_dir, **kwargs
        )
        video_results.vmaf_psnr(
            original_data_path, compressed_data_path, script_dir, **kwargs
        )
        video_results.ffmpeg_psnr(
            original_data_path, compressed_data_path, script_dir, **kwargs
        )
        video_results.ffmpeg_ssim(
            original_data_path, compressed_data_path, script_dir, **kwargs
        )
    else:
        base_path = ""
        compressed_data_path = os.path.join(base_path, kwargs.get("codec", "JPEG"))
        original_data_path = ""

        image_results.psnr(original_data_path, compressed_data_path, **kwargs)
        image_results.ms_ssim(original_data_path, compressed_data_path, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--directory",
        dest="directory",
        help="Path to run in",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-e",
        "--extension",
        dest="extension",
        help="Video Extension",
        default="mp4",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--c",
        dest="codec",
        help="Video codec being tested against. Default: H264",
        default="H264",
        type=str,
    )
    parser.add_argument(
        "-b" "--bitrate-raw",
        dest="codec_bitrate",
        help="Use bitrate of raw encoded video, rather than codec",
        default=True,
        action="store_false",
    )
    parser.add_argument(
        "-nd" "--no-deblock",
        dest="de_blocking",
        help="Turn off de-blocking in codec",
        default=True,
        action="store_false",
    )
    parser.add_argument(
        "-f",
        "--full",
        dest="full",
        action="store_true",
        help="Boolean - do a full run -- create 1080p video",
        default=False,
    )
    parser.add_argument(
        "-i",
        "--image",
        dest="video",
        action="store_false",
        help="Generate metrics for images (default is video)",
        default=True,
    )
    parser.add_argument(
        "-k",
        "--keep",
        dest="delete",
        action="store_false",
        help="Keep batch and data files from metric gathering",
        default=True,
    )
    parser.add_argument(
        "-m",
        "--motion",
        dest="motion_level",
        help="Specify level of motion; low, medium, high, default is all levels",
        default=None,
        type=str,
    )

    args = parser.parse_args()
    # Create a dictionary from the supplier args namespace
    args_dict = vars(args)

    script_dir = os.path.dirname(os.path.realpath(__file__))

    main(script_dir, **args_dict)
