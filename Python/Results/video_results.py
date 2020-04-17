#!/usr/bin/env python3
import os
import argparse
import results_util as util


def main(*args, **kwargs):
    base_path = ""
    compressed_data_path = os.path.join(base_path, kwargs.get("codec", "H264"))
    original_data_path = ""

    vmaf_score(original_data_path, compressed_data_path, *args, **kwargs)
    vmaf_psnr(original_data_path, compressed_data_path, *args, **kwargs)
    ffmpeg_psnr(original_data_path, compressed_data_path, *args, **kwargs)
    ffmpeg_ssim(original_data_path, compressed_data_path, *args, **kwargs)


def vmaf_score(
    original_data_path: str,
    compressed_data_path: str,
    script_dir: str,
    codec: str = "H264",
    motion_level: str = None,
    **kwargs,
):
    """
    Utilise VMAF library to compute VMAF metric
    :param original_data_path: original videos
    :param compressed_data_path: compressed videos (baseline for comparison)
    :param script_dir: Location bash script to use
    :param codec: Codec being tested
    :param motion_level: Levels of motion to use for VMAF score
    :param kwargs:
    :return:
    """
    metric = "VMAF"
    print(f"{codec} - {metric}")
    vmaf_script = os.path.join(script_dir, "vmaf_batch.sh")
    metrics, rate_metrics = util.vmaf_metric(
        original_data_path,
        compressed_data_path,
        vmaf_script,
        metric,
        **{**kwargs, **{"codec": codec, "motion": motion_level}},
    )

    if motion_level:
        folder_name = f"{motion_level}_{metric}"
    else:
        folder_name = None

    util.make_plots(metric, metrics, codec, folder_name=folder_name)
    util.make_plots(metric, rate_metrics, codec, rate=True, folder_name=folder_name)


def vmaf_psnr(
    original_data_path,
    compressed_data_path,
    script_dir,
    codec="H264",
    motion_level: str = None,
    **kwargs,
):
    """
    Utilise VMAF library to compute PSNR
    :param original_data_path: original videos
    :param compressed_data_path: compressed videos (baseline for comparison)
    :param script_dir: Location bash script to use
    :param codec: Codec being tested
    :param motion_level: Levels of motion to use for PSNR
    :param kwargs:
    :return:
    """
    metric = "PSNR"
    print(f"{codec} - {metric}")
    vmaf_script = os.path.join(script_dir, "psnr_batch.sh")
    metrics, rate_metrics = util.vmaf_metric(
        original_data_path,
        compressed_data_path,
        vmaf_script,
        metric,
        **{**kwargs, **{"codec": codec, "motion": motion_level}},
    )

    if motion_level:
        folder_name = f"{motion_level}_{metric}_VMAF"
    else:
        folder_name = f"{metric}_VMAF"

    util.make_plots(metric, metrics, codec, folder_name=folder_name)
    util.make_plots(metric, rate_metrics, codec, rate=True, folder_name=folder_name)


def ffmpeg_ssim(
    original_data_path,
    compressed_data_path,
    script_dir,
    codec="H264",
    motion_level: str = None,
    **kwargs,
):
    """
    Utilise FFmpeg library to compute SSIM
    :param original_data_path: original videos
    :param compressed_data_path: compressed videos (baseline for comparison)
    :param script_dir: Location bash script to use
    :param codec: Codec being tested
    :param motion_level: Levels of motion to use for SSIM
    :param kwargs:
    :return:
    """
    metric = "SSIM"
    print(f"{codec} - {metric}")
    metrics = util.ffmpeg_metric(
        original_data_path,
        compressed_data_path,
        script_dir,
        metric,
        **{**kwargs, **{"codec": codec, "motion": motion_level}},
    )

    if motion_level:
        folder_name = f"{motion_level}_{metric}"
    else:
        folder_name = f"{metric}"

    util.make_plots(metric, metrics, codec, rate=True, folder_name=folder_name)


def ffmpeg_psnr(
    original_data_path,
    compressed_data_path,
    script_dir,
    codec="H264",
    motion_level: str = None,
    **kwargs,
):
    """
    Utilise FFmpeg library to compute PSNR
    :param original_data_path: original videos
    :param compressed_data_path: compressed videos (baseline for comparison)
    :param script_dir: Location bash script to use
    :param codec: Codec being tested
    :param motion_level: Levels of motion to use for PSNR
    :param kwargs:
    :return:
    """
    metric = "PSNR"
    print(f"{codec} - {metric}")
    metrics = util.ffmpeg_metric(
        original_data_path,
        compressed_data_path,
        script_dir,
        metric,
        **{**kwargs, **{"codec": codec, "motion": motion_level}},
    )

    if motion_level:
        folder_name = f"{motion_level}_{metric}_FFMPEG"
    else:
        folder_name = f"{metric}_FFMPEG"

    util.make_plots(metric, metrics, codec, rate=True, folder_name=folder_name)


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
        default="avi",
        type=str,
    )

    args = parser.parse_args()

    p_dir = os.path.dirname(os.path.realpath(__file__))

    main(p_dir, directory=args.directory, extension=args.extension)
