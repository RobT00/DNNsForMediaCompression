#!/usr/bin/env python3
"""
Function for generating image specific metrics
"""
import os
import argparse
import results_util as util


def main(*args, **kwargs):
    compressed_data_path = ""
    original_data_path = ""
    psnr(original_data_path, compressed_data_path, *args, **kwargs)


def psnr(
    original_data_path: str,
    compressed_data_path: str,
    *args,
    codec: str = "JPEG",
    **kwargs,
):
    """
    Create image metrics for PSNR
    :param original_data_path: Path for label images
    :param compressed_data_path: Path for model input images
    :param args:
    :param codec: Codec used for input images
    :param kwargs:
    :return:
    """
    metric = "PSNR"
    print(f"{codec} - {metric}")
    metric_fn = util.calc_psnr
    metrics, rate_metrics = util.image_metrics(
        original_data_path, compressed_data_path, metric_fn, **kwargs
    )
    # print(f"{metric} @ 90%: {metrics['768x512']['trained']['90']}")
    # try:
    #     print(
    #         f"{metric} 0.3266 bpp: {rate_metrics['768x512']['trained'][0.3265889485677083]}"
    #     )
    # except KeyError:
    #     print(f"{metric} bpps: {rate_metrics['768x512']['trained']}")
    util.make_plots(metric, metrics, codec, video=False)
    util.make_plots(metric, rate_metrics, codec, video=False, rate=True)


def ms_ssim(
    original_data_path: str,
    compressed_data_path: str,
    *args,
    codec: str = "JPEG",
    **kwargs,
):
    """
    Create image metrics for MS-SSIM
    :param original_data_path: Path for label images
    :param compressed_data_path: Path for model input images
    :param args:
    :param codec: Codec used for input images
    :param kwargs:
    :return:
    """
    metric = "MS-SSIM"
    print(f"{codec} - {metric}")
    metric_fn = util.calc_ms_ssim
    metrics, rate_metrics = util.image_metrics(
        original_data_path, compressed_data_path, metric_fn, **kwargs
    )
    # print(f"{metric} @ 90%: {metrics['768x512']['trained']['90']}")
    # try:
    #     print(
    #         f"{metric} 0.3266 bpp: {rate_metrics['768x512']['trained'][0.3265889485677083]}"
    #     )
    # except KeyError:
    #     print(f"{metric} bpps: {rate_metrics['768x512']['trained']}")
    util.make_plots(metric, metrics, codec, video=False)
    util.make_plots(metric, rate_metrics, codec, video=False, rate=True)


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
        help="Image Extension",
        default="png",
        type=str,
    )

    args = parser.parse_args()

    p_dir = os.path.dirname(os.path.realpath(__file__))

    main(p_dir, directory=args.directory, extension=args.extension)
