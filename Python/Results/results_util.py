#!/usr/bin/env python3
"""
File containing utility functions
Author: Robert Trew
"""
import os
import shutil
import re
import glob
import cv2
import json
import numpy as np
import math
import pickle
from scipy import signal
from scipy import stats
from scipy.ndimage.filters import convolve
import subprocess as sp
import matplotlib.pyplot as plt

SAVE_EXTENSION = "mkv"
OVERWRITE = "-n"

LOW_MOTION = ["akiyo_cif", "bowing_cif"]
MEDIUM_MOTION = ["container_cif", "crew_cif", "hall_monitor_cif"]
HIGH_MOTION = [
    "coastguard_cif",
    "container_cif",
    "foreman_cif",
    "football_422_cif",
    "football_cif",
]
SPLIT_VIDEOS = {"low": LOW_MOTION, "medium": MEDIUM_MOTION, "high": HIGH_MOTION}


def get_codec_options(codec: str = "H264", de_blocking: bool = True, **kwargs) -> list:
    """
    Reutrn codec specific options for FFmpeg commands
    :param codec: used for encoding
    :param de_blocking: enable/disable deblocking
    :param kwargs:
    :return: list of codec options to use
    """
    codec_options = list()
    if codec.upper() == "H264":
        codec_options.extend(["-c:v", "libx264"])
        if not de_blocking:
            codec_options.extend(["-x264opts", "no-deblock"])
    elif codec.upper() == "H265":
        codec_options.extend(["-c:v", "libx265"])
    elif codec.upper() == "AV1":
        codec_options.extend(["-c:v", "libaom-av1"])
    elif codec.upper() == "VP9":
        codec_options.extend(["-c:v", "libvpx-vp9"])

    return codec_options


def create_files(
    original_data_path: str,
    compressed_data_path: str,
    bash_file: str,
    directory: str = None,
    extension: str = "avi",
    full: bool = False,
    delete: bool = True,
    motion_levels: list = None,
    **kwargs,
) -> dict:
    """
    Function to create files necessary for metric generation, utilising VMAF library
    :param original_data_path: Path for original files
    :param compressed_data_path: Path of input files
    :param bash_file: Bash script used for metrics
    :param directory: Directory to output files to
    :param extension: Extension of trained files
    :param full: Create 1080p representations
    :param delete: Clean up after running
    :param motion_levels: Levels of motion to use to metric generation
    :param kwargs:
    :return: Record metrics for plotting, average bitrate of each video representation
    """
    batch_location = "batch_files"
    data_location = "data_files"
    os.chdir(directory)
    crfs = glob.glob("crf=*")
    create_empty_dir(batch_location)
    create_empty_dir(data_location)
    avg_bitrate = dict()
    for q in crfs:
        if not os.path.isdir(q):
            continue
        print(q)
        qual = q.split("=")[-1]
        compressed_batch_input = list()
        compressed_bitrates_input = list()
        compressed_batch_input_original = list()
        compressed_bitrates_original = list()
        compressed_batch_input_1080 = list()
        compressed_bitrates_1080 = list()
        trained_batch_input = list()
        trained_bitrates_input = list()
        trained_batch_input_original = list()
        trained_bitrates_original = list()
        trained_batch_input_1080 = list()
        trained_bitrates_1080 = list()
        folders = glob.glob(q + f"{os.sep}*")
        for f in folders:
            if motion_levels:
                if not any(os.path.basename(f) == m for m in motion_levels):
                    print(f)
                    continue
            d, trained_resolution, original_resolution = create_videos(
                original_data_path,
                compressed_data_path,
                extension,
                f,
                os.path.dirname(bash_file),
                qual,
                full=full,
                **kwargs,
            )

            # yuv_fmt width height ref_path distort_path
            # Trained dims
            compressed_batch_input.append(
                f"yuv420p {trained_resolution['width']} {trained_resolution['height']} "
                f"{d['original_' + trained_resolution['height']]['path']} "
                f"{d['compressed_' + trained_resolution['height']]['path']}\n"
            )
            compressed_bitrates_input.append(
                d["compressed_" + trained_resolution["height"]]["bitrate"]
            )
            trained_batch_input.append(
                f"yuv420p {trained_resolution['width']} {trained_resolution['height']} "
                f"{d['original_' + trained_resolution['height']]['path']} "
                f"{d['trained_' + trained_resolution['height']]['path']}\n"
            )
            trained_bitrates_input.append(
                d["trained_" + trained_resolution["height"]]["bitrate"]
            )

            # Original dims
            compressed_batch_input_original.append(
                f"yuv420p {original_resolution['width']} {original_resolution['height']} "
                f"{d['original_' + original_resolution['height']]['path']} "
                f"{d['compressed_' +  original_resolution['height']]['path']}\n"
            )
            compressed_bitrates_original.append(
                d["compressed_" + original_resolution["height"]]["bitrate"]
            )
            trained_batch_input_original.append(
                f"yuv420p {original_resolution['width']} {original_resolution['height']} "
                f"{d['original_' + original_resolution['height']]['path']} "
                f"{d['trained_' +  original_resolution['height']]['path']}\n"
            )
            trained_bitrates_original.append(
                d["trained_" + original_resolution["height"]]["bitrate"]
            )

            # 1080
            if full:
                compressed_batch_input_1080.append(
                    f"yuv420p 1920 1080 {d['original_1080']['path']} {d['compressed_1080']['path']}\n"
                )
                compressed_bitrates_1080.append(d["compressed_1080"]["bitrate"])
                trained_batch_input_1080.append(
                    f"yuv420p 1920 1080 {d['original_1080']['path']} {d['trained_1080']['path']}\n"
                )
                trained_bitrates_1080.append(d["trained_1080"]["bitrate"])

        print("running scripts")
        # fmt: off
        # Trained dims
        with open(
            f"{batch_location}{os.sep}compressed_batch_input_{trained_resolution['height']}",
            "w+",
        ) as f_in:
            f_in.writelines(compressed_batch_input)

        vmaf_call = [
            "sh",
            bash_file,
            f"{batch_location}{os.sep}compressed_batch_input_{trained_resolution['height']}",
            f"{data_location}{os.sep}{q}-compressed_batch_input-{trained_resolution['height']}",
        ]
        # fmt: on
        sp.call(vmaf_call)
        # fmt: off
        with open(
            f"{batch_location}{os.sep}trained_batch_input_{trained_resolution['height']}",
            "w+",
        ) as f_in:
            f_in.writelines(trained_batch_input)

        vmaf_call = [
            "sh", bash_file,
            f"{batch_location}{os.sep}trained_batch_input_{trained_resolution['height']}",
            f"{data_location}{os.sep}{q}-trained_batch_input-{trained_resolution['height']}",
        ]
        # fmt: on
        sp.call(vmaf_call)
        # fmt: off
        # Original dims
        with open(
            f"{batch_location}{os.sep}compressed_batch_input_{original_resolution['height']}",
            "w+",
        ) as f_in:
            f_in.writelines(compressed_batch_input_original)

        vmaf_call = [
            "sh", bash_file,
            f"{batch_location}{os.sep}compressed_batch_input_{original_resolution['height']}",
            f"{data_location}{os.sep}{q}-compressed_batch_input-{original_resolution['height']}",
        ]
        # fmt: on
        sp.call(vmaf_call)
        # fmt: off
        with open(
            f"{batch_location}{os.sep}trained_batch_input_{original_resolution['height']}",
            "w+",
        ) as f_in:
            f_in.writelines(trained_batch_input_original)

        vmaf_call = [
            "sh", bash_file,
            f"{batch_location}{os.sep}trained_batch_input_{original_resolution['height']}",
            f"{data_location}{os.sep}{q}-trained_batch_input-{original_resolution['height']}",
        ]
        # fmt: on
        sp.call(vmaf_call)

        # 1080
        if full:
            # fmt: off
            with open(
                f"{batch_location}{os.sep}compressed_batch_input_1080",
                "w+"
            ) as f_in:
                f_in.writelines(compressed_batch_input_1080)

            vmaf_call = [
                "sh", bash_file,
                f"{batch_location}{os.sep}compressed_batch_input_1080",
                f"{data_location}{os.sep}{q}-compressed_batch_input-1080",
            ]
            # fmt: on
            sp.call(vmaf_call)
            # fmt: off
            with open(
                f"{batch_location}{os.sep}trained_batch_input_1080",
                "w+"
            ) as f_in:
                f_in.writelines(trained_batch_input_1080)

            vmaf_call = [
                "sh", bash_file,
                f"{batch_location}{os.sep}trained_batch_input_1080",
                f"{data_location}{os.sep}{q}-trained_batch_input-1080",
            ]
            # fmt: on
            sp.call(vmaf_call)

        avg_bitrate.setdefault(trained_resolution["height"], dict()).setdefault(
            "trained", dict()
        ).update({qual: sum(trained_bitrates_input) / len(trained_bitrates_input)})
        avg_bitrate.setdefault(trained_resolution["height"], dict()).setdefault(
            "compressed", dict()
        ).update(
            {qual: sum(compressed_bitrates_input) / len(compressed_bitrates_input)}
        )

        avg_bitrate.setdefault(original_resolution["height"], dict()).setdefault(
            "trained", dict()
        ).update(
            {qual: sum(trained_bitrates_original) / len(trained_bitrates_original)}
        )
        avg_bitrate.setdefault(original_resolution["height"], dict()).setdefault(
            "compressed", dict()
        ).update(
            {
                # fmt: off
                qual: sum(compressed_bitrates_original) / len(compressed_bitrates_original)
                # fmt: on
            }
        )
        if full:
            avg_bitrate.setdefault("1080", dict()).setdefault("trained", dict()).update(
                {qual: sum(trained_bitrates_1080) / len(trained_bitrates_1080)}
            )
            avg_bitrate.setdefault("1080", dict()).setdefault(
                "compressed", dict()
            ).update(
                {qual: sum(compressed_bitrates_1080) / len(compressed_bitrates_1080)}
            )
    if delete:
        delete_folder(batch_location)
    return avg_bitrate


def create_videos(
    original_data_path: str,
    compressed_data_path: str,
    extension: str,
    folder: str,
    script_dir: str,
    qual: str,
    full: bool = False,
    codec_bitrate: bool = False,
    **kwargs,
) -> tuple:
    """
    Create video representations with FFmpeg, utilised for metric gathering
    :param original_data_path: Original video path
    :param compressed_data_path: Compressed video path
    :param extension: File extension to create representations from
    :param folder: Folder containing video representations (is also name of video)
    :param script_dir: Bash script utilised for gathering bitrate
    :param qual: CRF of video
    :param full: Create 1080p representation
    :param codec_bitrate: Capture bitrate using specified codec
    :param kwargs:
    :return: Video metadata, resolution model trained at, resolution of original video
    """
    vids = dict()
    vids["trained"] = os.path.join(folder, f"trained.{extension}")
    trained_resolution = video_resolution(vids["trained"])
    vids["original"] = os.path.join(folder, f"original.{extension}")
    original = os.path.join(original_data_path, f"{folder.split(os.sep)[-1]}.y4m")
    original_resolution = video_resolution(original)
    compressed = os.path.join(
        compressed_data_path, f"{folder.split(os.sep)[-1]}_{qual}.mp4"
    )
    vids["compressed"] = os.path.join(folder, f"compressed.{extension}")
    d = dict()
    for vid_name, vid_path in vids.items():
        # Convert file to yuv420p
        v_trained = os.path.join(folder, vid_name + f"_{trained_resolution['height']}")
        # fmt: off
        ffmpeg_cmd = [
            "ffmpeg",
            OVERWRITE,
            "-v", "error",
            "-i", vid_path,
            "-pix_fmt", "yuv420p",
            "-c:v", "rawvideo",
            # "-vf", f"scale={trained_resolution['width']}:{trained_resolution['height']}:"
            #        f"force_original_aspect_ratio=decrease",
            "-vf", f"scale={trained_resolution['width']}:{trained_resolution['height']}",
            "-sws_flags", "bicubic",
            f"{v_trained}.{SAVE_EXTENSION}"
        ]
        # fmt: on
        sp.call(ffmpeg_cmd)
        if codec_bitrate:
            # fmt: off
            ffmpeg_cmd = [
                "ffmpeg",
                OVERWRITE,
                "-v", "error",
                "-i", vid_path,
                # "-map_metadata", "-1",
                "-crf", qual,
                # "-preset", "veryslow",
                # "-profile:v", "main",
                "-pix_fmt", "yuv420p",
                # "-movflags", "+faststart",
                "-vf", f"scale={trained_resolution['width']}:{trained_resolution['height']}",
                # "-vf", f"scale={trained_resolution['width']}:{trained_resolution['height']}:"
                #        f"force_original_aspect_ratio=decrease",
                "-sws_flags", "bicubic",
            ]
            # fmt: on
            ffmpeg_cmd.extend(get_codec_options(**kwargs))
            ffmpeg_cmd.append(f"{v_trained}.mp4")
            sp.call(ffmpeg_cmd)
            bitrate = video_bitrate(script_dir, f"{v_trained}.mp4")
        else:
            bitrate = video_bitrate(script_dir, f"{v_trained}.{SAVE_EXTENSION}")
        d.update(
            {
                # fmt: off
                vid_name + f"_{trained_resolution['height']}": {
                    # fmt: on
                    "path": f"{v_trained}.{SAVE_EXTENSION}",
                    "width": trained_resolution["width"],
                    "height": trained_resolution["height"],
                    "bitrate": bitrate,
                }
            }
        )

        # Create yuv420p file for source dimensions
        v_original = os.path.join(
            folder, vid_name + f"_{original_resolution['height']}"
        )
        # fmt: off
        ffmpeg_cmd = [
            "ffmpeg",
            OVERWRITE,
            "-v", "error",
            "-i", vid_path,
            "-pix_fmt", "yuv420p",
            "-c:v", "rawvideo",
            # "-vf", f"scale={original_resolution['width']}:{original_resolution['height']}",
            "-vf", f"scale={original_resolution['width']}:{original_resolution['height']}:"
                   f"force_original_aspect_ratio=decrease",
            "-sws_flags", "lanczos",
            f"{v_original}.{SAVE_EXTENSION}"
        ]
        sp.call(ffmpeg_cmd)
        if codec_bitrate:
            ffmpeg_cmd = [
                "ffmpeg",
                OVERWRITE,
                "-v", "error",
                "-i", vid_path,
                # "-map_metadata", "-1",
                "-crf", qual,
                # "-preset", "veryslow",
                # "-profile:v", "main",
                # "-movflags", "+faststart"
                "-pix_fmt", "yuv420p",
                # "-vf", f"scale={original_resolution['width']}:{original_resolution['height']}",
                "-vf", f"scale={original_resolution['width']}:{original_resolution['height']}:"
                       f"force_original_aspect_ratio=decrease",
                "-sws_flags", "lanczos",
            ]
            ffmpeg_cmd.extend(get_codec_options(**kwargs))
            ffmpeg_cmd.append(f"{v_original}.mp4")
            sp.call(ffmpeg_cmd)
            bitrate = video_bitrate(script_dir, f"{v_original}.mp4")
        else:
            bitrate = video_bitrate(script_dir, f"{v_original}.{SAVE_EXTENSION}")
        d.update(
            {
                vid_name + f"_{original_resolution['height']}": {
                    "path": f"{v_original}.{SAVE_EXTENSION}",
                    "width": original_resolution['width'],
                    "height": original_resolution['height'],
                    "bitrate": bitrate
                }
            }
        )
        # fmt: on
        if full:
            # Create yuv420p file at 1080p
            v_1080 = os.path.join(folder, vid_name + "_1080")
            # fmt: off
            ffmpeg_cmd = [
                "ffmpeg",
                OVERWRITE,
                "-v", "error",
                "-i", vid_path,
                "-pix_fmt", "yuv420p",
                "-c:v", "rawvideo",
                "-vf", "scale=1920:1080:"
                       "force_original_aspect_ratio=decrease",
                "-sws_flags", "lanczos",
                f"{v_1080}.{SAVE_EXTENSION}"
            ]
            sp.call(ffmpeg_cmd)
            if codec_bitrate:
                # fmt: off
                ffmpeg_cmd = [
                    "ffmpeg",
                    OVERWRITE,
                    "-v", "error",
                    "-i", vid_path,
                    # "-map_metadata", "-1",
                    "-crf", qual,
                    # "-preset", "veryslow",
                    # "-profile:v", "main",
                    "-pix_fmt", "yuv420p",
                    # "-movflags", "+faststart"
                    "-vf", "scale=1920:1080:"
                           "force_original_aspect_ratio=decrease",
                    "-sws_flags", "lanczos",
                ]
                # fmt: on
                ffmpeg_cmd.extend(get_codec_options(**kwargs))
                ffmpeg_cmd.append(f"{v_1080}.mp4")
                sp.call(ffmpeg_cmd)
                bitrate = video_bitrate(script_dir, f"{v_1080}.mp4")
            else:
                bitrate = video_bitrate(script_dir, f"{v_1080}.{SAVE_EXTENSION}")
            d.update(
                {
                    vid_name + "_1080": {
                        "path": f"{v_1080}.{SAVE_EXTENSION}",
                        "width": "1920",
                        "height": "1080",
                        "bitrate": bitrate
                    }
                }
            )
            # fmt: on

    return d, trained_resolution, original_resolution


def json_parser(
    score: str = "VMAF", delete: bool = True, rates: dict = None, **kwargs
) -> tuple:
    """
    Parse JSON files to calulate metric scores
    :param score: Metric to gather
    :param delete: Clean up after executing
    :param rates: Gather bitrates
    :param kwargs:
    :return: Aggregate metrics for video CRF, aggregate metrics for video bitrates
    """
    score = "_".join([score, "score"])
    jsons = glob.glob(f"*/*.json")
    d = dict()
    rd = dict()
    for j in jsons:
        scores = list()
        reg_exp = re.compile(r"\d+\.?\d+")
        search_score = False
        # JSON seems poorly formatted, this will do
        for line in open(j, "r"):
            if not search_score and "aggregate" in line.lower():
                search_score = True
            if search_score and score.upper() in line.upper():
                scores.append(float(reg_exp.findall(line)[0]))
                search_score = False

        j = j.split("-")
        q = j.pop(0)
        q = q.split("=")[-1]
        r = re.findall(r"\d+", j[-1])[0]
        c = j[0].split("_")[0]
        try:
            # score_avg = sum(scores) / len(scores)
            score_avg = stats.hmean(scores)
        except ZeroDivisionError:
            score_avg = 0.0
        d.setdefault(r, dict()).setdefault(c, dict()).update({q: score_avg})
        if rates is not None:
            rate = rates[r][c][q]
            rd.setdefault(r, dict()).setdefault(c, dict()).update({rate: score_avg})
    if delete:
        delete_folder(jsons[0], up=True)
    return d, rd


def vmaf_metric(
    original_data_path: str,
    compressed_data_path: str,
    vmaf_script: str,
    metric: str,
    motion: str = None,
    **kwargs,
) -> tuple:
    """
    Base function for gathering metrics using the VMAF library
    :param original_data_path: Path to original videos
    :param compressed_data_path: Path to compressed videos
    :param vmaf_script: Directory of script leveraging VMAF functionality for metric generation
    :param metric: Metric to gather
    :param motion: Levels of motion to gather metrics for
    :param kwargs:
    :return: Aggreate metrics for CRF, Aggregate metrics for bitrate
    """
    # Make files, generate batch files, generate json
    motion_level = SPLIT_VIDEOS.get(motion)
    rates = create_files(
        original_data_path,
        compressed_data_path,
        vmaf_script,
        motion_levels=motion_level,
        **kwargs,
    )
    # Parse json
    metrics, rate_metrics = json_parser(metric, rates=rates, **kwargs)

    return metrics, rate_metrics


def ffmpeg_metric(
    org_dir,
    comp_dir,
    script_dir,
    metric,
    directory=None,
    extension: str = "mp4",
    full: bool = False,
    motion: str = None,
    **kwargs,
) -> dict:
    """
    Base function for gathering metrics using FFmpeg
    :param org_dir: Path to original videos
    :param comp_dir: Path to compressed videos
    :param script_dir: Directory of script leveraging FFmpeg functionality for metric generation
    :param metric: Metric to gather
    :param directory: Directory to store created video representations in
    :param extension: Extension of videos to create representations from
    :param full: Create 1080p representations
    :param motion: Specified levels of motion to gather metrics for
    :param kwargs:
    :return: Metrics aggregated on bitrate
    """
    metrics = dict()
    motion_level = SPLIT_VIDEOS.get(motion)
    os.chdir(directory)
    crfs = glob.glob("crf=*")
    for q in crfs:
        if not os.path.isdir(q):
            continue
        print(q)
        qual = q.split("=")[-1]
        folders = glob.glob(q + f"{os.sep}*")
        t_metric_score = dict()
        t_bit_rate = dict()
        c_metric_score = dict()
        c_bit_rate = dict()
        for f in folders:
            if motion_level:
                if not (any(os.path.basename(f) == m for m in motion_level)):
                    print(f)
                    continue
            d, trained_resolution, original_resolution = create_videos(
                org_dir, comp_dir, extension, f, script_dir, qual, full=full, **kwargs
            )
            resolutions = [trained_resolution["height"], original_resolution["height"]]
            if full:
                resolutions.append("1080")
            for res in resolutions:
                t_vid = d[f"trained_{res}"]["path"]
                o_vid = d[f"original_{res}"]["path"]
                c_vid = d[f"compressed_{res}"]["path"]
                # fmt: off
                ffmpeg_cmd = [
                    os.path.join(script_dir, "ffmpeg.sh"),
                    metric.lower(),
                    t_vid,
                    o_vid,
                ]
                # fmt: on
                sp.call(ffmpeg_cmd)
                t_metric_score.setdefault(res, list()).append(
                    text_parser(f"{metric}.txt", metric)
                )
                os.remove(f"{metric}.txt")
                t_bit_rate.setdefault(res, list()).append(
                    d[f"trained_{res}"]["bitrate"]
                )
                # fmt: off
                ffmpeg_cmd = [
                    os.path.join(script_dir, "ffmpeg.sh"),
                    metric.lower(),
                    c_vid,
                    o_vid,
                ]
                # fmt: on
                sp.call(ffmpeg_cmd)
                c_metric_score.setdefault(res, list()).append(
                    text_parser(f"{metric}.txt", metric)
                )
                os.remove(f"{metric}.txt")
                # c_bit_rate.setdefault(res, list()).append(
                #     d[f"compressed_{res}"]["bitrate"]
                # )
                # Using original because we are compressing the original to get our bitrates
                # More accurate than re-encoded file
                c_bit_rate.setdefault(res, list()).append(
                    d[f"original_{res}"]["bitrate"]
                )

        for res in resolutions:
            # t_metric_score[res] = sum(t_metric_score[res]) / len(t_metric_score[res])
            t_metric_score[res] = stats.hmean(t_metric_score[res])
            # c_metric_score[res] = sum(c_metric_score[res]) / len(c_metric_score[res])
            c_metric_score[res] = stats.hmean(c_metric_score[res])

            t_bit_rate[res] = sum(t_bit_rate[res]) / len(t_bit_rate[res])
            c_bit_rate[res] = sum(c_bit_rate[res]) / len(c_bit_rate[res])

            # metrics.setdefault(res, dict()).setdefault("trained", dict()).update(
            #     {c_bit_rate[res]: t_metric_score[res]}
            # )
            metrics.setdefault(res, dict()).setdefault("trained", dict()).update(
                {t_bit_rate[res]: t_metric_score[res]}
            )
            metrics.setdefault(res, dict()).setdefault("compressed", dict()).update(
                {c_bit_rate[res]: c_metric_score[res]}
            )
    return metrics


def image_metrics(
    original_data_path: str,
    compressed_data_path: str,
    metric_fn,
    directory: str = None,
    extension: str = "png",
    **kwargs,
) -> tuple:
    """
    Gather metrics on image model's performance
    :param original_data_path: Original images
    :param compressed_data_path: Compressed Images
    :param metric_fn: Specific metrics to gather
    :param directory: Directory to store data in
    :param extension: Extension of images to gather metrics around
    :param kwargs:
    :return: Aggregated metrics on compression level, Aggregated metrics on bpp
    """
    d_q = dict()
    d_bpp = dict()
    res = -1
    reg_exp = re.compile(r"\d+$")
    if directory is not None:
        os.chdir(directory)
    qualties = glob.glob(r"*")
    for q in qualties:
        if not reg_exp.match(q):
            continue
        ims = glob.glob(f"{q}/*")
        compressed_scores = list()
        trained_scores = list()
        compressed_bpps = list()
        trained_bpps = list()
        for im in ims:
            original_im = os.path.join(
                original_data_path, f"{im.split(os.sep)[-1]}.png"
            )
            original = cv2.imread(original_im)
            compressed_im = os.path.join(
                compressed_data_path, f"{im.split(os.sep)[-1]}_{q}.jpg"
            )
            compressed = cv2.imread(compressed_im)
            trained_im = os.path.join(im, f"trained.{extension}")
            trained = cv2.imread(trained_im)

            if original.shape[0] == trained.shape[1]:
                trained = np.rot90(trained, 3)

            # if res != get_res(original):
            if res == -1:
                res = image_resolution(original)
                d_q.setdefault(res, dict())
                d_bpp.setdefault(res, dict())
            if trained.shape[0] == trained.shape[1]:
                # Image is square, produced from video model
                t_res = image_resolution(original)
                trained = cv2.resize(
                    trained,
                    (int(t_res.split("x")[0]), int(t_res.split("x")[1])),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            max_pixel = original.max()
            if max_pixel != 255:
                print(max_pixel)
            original = np.expand_dims(original, axis=0)
            compressed = np.expand_dims(compressed, axis=0)
            trained = np.expand_dims(trained, axis=0)
            compressed_score = metric_fn(compressed, original, max_val=max_pixel)
            trained_score = metric_fn(trained, original, max_val=max_pixel)

            compressed_scores.append(compressed_score)
            trained_scores.append(trained_score)

            compressed_bpps.append(image_entropy(compressed_im))
            trained_bpps.append(image_entropy(trained_im))

        # avg_compressed = sum(compressed_scores) / len(compressed_scores)
        avg_compressed = stats.hmean(compressed_scores)
        # avg_trained = sum(trained_scores) / len(trained_scores)
        avg_trained = stats.hmean(trained_scores)

        d_q[res].setdefault("compressed", dict()).update({q: avg_compressed})
        d_q[res].setdefault("trained", dict()).update({q: avg_trained})

        bpp_c = sum(compressed_bpps) / len(compressed_bpps)
        bpp_t = sum(trained_bpps) / len(trained_bpps)

        d_bpp[res].setdefault("compressed", dict()).update({bpp_c: avg_compressed})
        d_bpp[res].setdefault("trained", dict()).update({bpp_c: avg_trained})

    return d_q, d_bpp


def text_parser(text_file: str, metric: str, **kwargs) -> float:
    """
    Parse FFmpeg generated text file to gather metrics
    :param text_file: Path to metric file
    :param metric: Metric to gather
    :param kwargs:
    :return: Harmonic mean of specified metric
    """
    score = list()
    with open(text_file, "r") as t:
        lines = t.readlines()

    if metric.lower() == "ssim":
        search_string = "all"
    else:
        search_string = f"{metric.lower()}_avg"

    for line in lines:
        score.append(
            float(
                [i for i in line.split(" ") if search_string in i.lower()][0].split(
                    ":"
                )[1]
            )
        )
        # score += float(
        #     [i for i in line.split(" ") if search_string in i.lower()][0].split(":")[1]
        # )

    # return score / len(lines)
    return float(stats.hmean(score))


def make_plots(
    metric_name: str,
    metric_dict: dict,
    codec: str,
    video: bool = True,
    rate: bool = False,
    folder_name: str = None,
):
    """
    Create and save plots to visualise created metric dictionaries
    :param metric_name: Name of metric being recorded
    :param metric_dict: Dictionary of aggregated metrics to create plot from
    :param codec: Codec being assessed with regards to metrics
    :param video: Specify vide/image metrics
    :param rate: Specify is metric is quality level or bitrate (bpp)
    :param folder_name: Override folder name for saving to
    :return:
    """
    os.makedirs("PLOTS", exist_ok=True)
    folder_name = folder_name if folder_name is not None else metric_name
    if rate:
        plot_name = unique_file(f"rate_{folder_name.lower()}", "PLOTS")
    else:
        plot_name = unique_file(folder_name.lower(), "PLOTS")
    os.makedirs(plot_name)
    print(plot_name)
    codec_colour = {"compressed": "blue", "trained": "red"}
    codec_name = {"compressed": "encoded", "trained": "post-processed"}
    codec_marker = {"compressed": "x", "trained": "o"}
    for res, encoders in metric_dict.items():
        fig = plt.figure()
        for encoder, data in encoders.items():
            qualities = list()
            scores = list()
            for qual, score in data.items():
                qualities.append(qual)
                scores.append(score)
            z = sorted(zip(qualities, scores), key=lambda x: x[0], reverse=not (rate))
            qualities, scores = zip(*z)
            plt.plot(
                qualities,
                scores,
                label=f"{codec_name[encoder]}",
                color=codec_colour[encoder],
                marker=codec_marker[encoder],
            )
        if video:
            if rate:
                plt.xlabel("Rate (Mbps)")
            else:
                plt.xlabel("Quality (Constant Rate Factor)")
        else:
            if rate:
                plt.xlabel("Rate (bpp)")
            else:
                plt.xlabel(f"Quality ({codec} Compression [%])")
        if metric_name.upper() == "PSNR":
            plt.ylabel(f"{metric_name.upper()} [dB]")
        else:
            plt.ylabel(metric_name.upper())
        plt.legend()
        plt.title(f"{codec} - {metric_name.upper()} Scores at {res}p")

        save_name = f"{plot_name}{os.sep}{metric_name.lower()}_{res}"
        fig.savefig(f"{save_name}.png")

        with open(f"{save_name}.pickle", "wb") as pkl:
            pickle.dump(encoders, pkl, protocol=pickle.HIGHEST_PROTOCOL)


def video_resolution(vid: str) -> dict:
    """
    Return resolution of a video via FFmpeg
    :param vid: Path to video
    :return: height and width of video (in pixels)
    """
    # fmt: off
    ffprobe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "default=nw=1:nk=1",
        vid,
    ]
    # fmt: on
    rc = sp.run(ffprobe_cmd, stdout=sp.PIPE)
    width, height = map(int, re.findall(r"\d+", rc.stdout.decode()))

    return {"width": str(width), "height": str(height)}


def image_resolution(img: np.ndarray) -> str:
    """
    Return resoltuion of an image
    :param img: Ndarray of image
    :return: String of image width x height
    """
    return f"{img.shape[1]}x{img.shape[0]}"


def calc_psnr(img: np.ndarray, src_img: np.ndarray, max_val: float = 255.0):
    """
    Calculate PSNR for an image
    :param img: Distorted images to calulate PSNR for
    :param src_img: Original image
    :param max_val: Max pixel value in images
    :return: PSNR
    """
    mse = np.mean((img - src_img) ** 2)
    if mse == 0:
        val = 100
    else:
        val = 20 * math.log10(max_val / math.sqrt(mse))
    return val


# From https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py
# Adapted to remove need for loading TensorFlow
def calc_ms_ssim(
    img: np.ndarray,
    src_img: np.ndarray,
    max_val: float = 255.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    weights: list = None,
) -> np.ndarray:
    """Return the MS-SSIM score between `img` and `src_img`.
    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    Arguments:
      img: Numpy array holding the first RGB image batch, distorted image.
      src_img: Numpy array holding the second RGB image batch, source (original) image.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
      weights: List of weights for each level; if none, use five levels and the
        weights from the original paper.
    Returns:
      MS-SSIM score between `img1` and `img2`.
    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """

    def _FSpecialGauss(size: int, sigma: float):
        """Function to mimic the 'fspecial' gaussian MATLAB function."""
        radius = size // 2
        offset = 0.0
        start, stop = -radius, radius + 1
        if size % 2 == 0:
            offset = 0.5
            stop -= 1
        x, y = np.mgrid[offset + start : stop, offset + start : stop]
        assert len(x) == size
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / g.sum()

    def _SSIMForMultiScale(
        img1: np.ndarray,
        img2: np.ndarray,
        max_val: float = 255,
        filter_size: int = 11,
        filter_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        """Return the Structural Similarity Map between `img1` and `img2`.
        This function attempts to match the functionality of ssim_index_new.m by
        Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
        Arguments:
          img1: Numpy array holding the first RGB image batch.
          img2: Numpy array holding the second RGB image batch.
          max_val: the dynamic range of the images (i.e., the difference between the
            maximum the and minimum allowed values).
          filter_size: Size of blur kernel to use (will be reduced for small images).
          filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
            for small images).
          k1: Constant used to maintain stability in the SSIM calculation (0.01 in
            the original paper).
          k2: Constant used to maintain stability in the SSIM calculation (0.03 in
            the original paper).
        Returns:
          Pair containing the mean SSIM and contrast sensitivity between `img1` and
          `img2`.
        Raises:
          RuntimeError: If input images don't have the same shape or don't have four
            dimensions: [batch_size, height, width, depth].
        """
        if img1.shape != img2.shape:
            raise RuntimeError(
                "Input images must have the same shape (%s vs. %s).",
                img1.shape,
                img2.shape,
            )
        if img1.ndim != 4:
            raise RuntimeError(
                "Input images must have four dimensions, not %d", img1.ndim
            )

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        _, height, width, _ = img1.shape

        # Filter size can't be larger than height or width of images.
        size = min(filter_size, height, width)

        # Scale down sigma if a smaller filter size is used.
        sigma = size * filter_sigma / filter_size if filter_size else 0

        if filter_size:
            window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
            mu1 = signal.fftconvolve(img1, window, mode="valid")
            mu2 = signal.fftconvolve(img2, window, mode="valid")
            sigma11 = signal.fftconvolve(img1 * img1, window, mode="valid")
            sigma22 = signal.fftconvolve(img2 * img2, window, mode="valid")
            sigma12 = signal.fftconvolve(img1 * img2, window, mode="valid")
        else:
            # Empty blur kernel so no need to convolve.
            mu1, mu2 = img1, img2
            sigma11 = img1 * img1
            sigma22 = img2 * img2
            sigma12 = img1 * img2

        mu11 = mu1 * mu1
        mu22 = mu2 * mu2
        mu12 = mu1 * mu2
        sigma11 -= mu11
        sigma22 -= mu22
        sigma12 -= mu12

        # Calculate intermediate values used by both ssim and cs_map.
        c1 = (k1 * max_val) ** 2
        c2 = (k2 * max_val) ** 2
        v1 = 2.0 * sigma12 + c2
        v2 = sigma11 + sigma22 + c2
        ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
        cs = np.mean(v1 / v2)
        return ssim, cs

    if img.shape != src_img.shape:
        raise RuntimeError(
            "Input images must have the same shape (%s vs. %s).",
            img.shape,
            src_img.shape,
        )
    if img.ndim != 4:
        raise RuntimeError("Input images must have four dimensions, not %d", img.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img, src_img]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2,
        )
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [
            convolve(im, downsample_filter, mode="reflect") for im in [im1, im2]
        ]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
    return np.prod(mcs[0 : levels - 1] ** weights[0 : levels - 1]) * (
        mssim[levels - 1] ** weights[levels - 1]
    )


def video_bitrate(script_dir: str, vid: str) -> float:
    """
    Calculate bitrate of a video in Mbps using FFmpeg
    :param script_dir: FFmpeg script to invoke
    :param vid: Path to video
    :return: Bitrate in Mbps
    """
    division = 1e6  # Mbps
    out_file = "out_file"
    # fmt: off
    ffprobe_cmd = [
        os.path.join(script_dir, "ffprobe.sh"),
        vid,
        out_file
    ]
    # fmt: on
    sp.call(ffprobe_cmd)

    with open(f"{out_file}.json", "r") as j:
        json_data = json.load(j)
    bit_rate_format = float(json_data["format"]["bit_rate"])
    bit_rate_stream = None
    for stream_dict in json_data["streams"]:
        if stream_dict.get("codec_type", "").lower() == "video":
            bit_rate_stream = float(stream_dict["bit_rate"])
            break

    try:
        bit_rate = min(bit_rate_format, bit_rate_stream)
    except TypeError:
        bit_rate = bit_rate_format

    os.remove(f"{out_file}.json")

    return bit_rate / division


def image_entropy(img: str) -> float:
    """
    Return bits per pixel (bpp) of an image
    :param img: Image path
    :return: bpp
    """
    im_size = os.path.getsize(img) * 8  # record in bits
    img = cv2.imread(img)
    pixels = img.shape[0] * img.shape[1]
    entropy = im_size / pixels
    # entropy = 0.0
    # channels = np.arange(img.shape[2])
    # for channel in channels:
    #     chan_img = np.copy(img[:, :, channel])
    #     probs, _ = np.histogram(chan_img, density=True, bins=256)
    #
    #     entropy -= (probs * np.ma.log2(probs)).sum()
    # entropy /= img.shape[2]

    # probs, _ = np.histogram(img, density=True, bins=256)
    # entropy = -(probs * np.ma.log2(probs)).sum()

    return entropy


def concat_images_horizontal(
    imga: np.ndarray, imgb: np.ndarray, gap: int = 0, video: bool = False
) -> np.ndarray:
    """
    Combines two color image ndarrays horizontally
    :param imga: First image, to be concatenated
    :param imgb: Second image, to concatenate
    :param gap: Gap between images
    :param video: Specify if part of a video or still image
    :return: Concatenated images
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb + gap
    if video:
        new_img = np.zeros(shape=(max_height, total_width, 3))
    else:
        new_img = np.ones(shape=(max_height, total_width, 3))
    # Set all white
    new_img *= 255
    new_img[:ha, :wa] = imga
    new_img[:hb, wa + gap : total_width] = imgb

    return new_img


def concat_images_vertical(
    imga: np.ndarray, imgb: np.ndarray, gap: int = 0, video: bool = False
):
    """
    Combines two color image ndarrays vertically
    :param imga: First image, to be concatenated
    :param imgb: Second image, to concatenate
    :param gap: Gap between images
    :param video: Specify if part of a video or still image
    :return: Concatenated images
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_width = np.max([wa, wb])
    total_height = ha + hb + gap
    if video:
        new_img = np.zeros(shape=(total_height, max_width, 3))
    else:
        new_img = np.ones(shape=(total_height, max_width, 3))
    # Set all white
    new_img *= 255
    new_img[:ha, :wa] = imga
    new_img[ha + gap : total_height, :wb] = imgb

    return new_img


def concat_n_images(
    image_path_list: list,
    horizontal: bool = True,
    img_corner: str = "bottom_left",
    gap: int = 0,
    text: bool = True,
) -> np.ndarray:
    """
    Concatenate N color images from a list of image paths.
    :param image_path_list: List of image paths for concatenating
    :param horizontal: Concatenate horiziontally or vertically
    :param img_corner: Corner of image to overlay text in
    :param gap: Gap between images
    :param text: To overlay text
    :return: Concatenated images, with appropriate overlaid text (if required)
    """
    output = None
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.5
    fontColor = (255, 255, 255)
    thickness = 4
    lineType = 2
    corner_pos = (0, 0)
    for i, img_path in enumerate(image_path_list):
        img = cv2.imread(img_path)[:, :, :3]
        im_name = str(os.path.basename(img_path).split(".")[0])
        im_name = im_name[0].upper() + im_name[1:]
        if im_name == "Trained":
            im_name = "Post-Processed"
        if i == 0:
            corner_pos = get_image_corner(img, corner=img_corner, gap=gap)
            output = img
        else:
            if horizontal:
                output = concat_images_horizontal(output, img, gap=gap)
                corner_pos = (corner_pos[0] + img.shape[1] + gap, corner_pos[1])
            else:
                output = concat_images_vertical(output, img, gap=gap)
                corner_pos = (corner_pos[0], corner_pos[1] + img.shape[0] + gap)
        if text:
            cv2.putText(
                output,
                im_name,
                corner_pos,
                font,
                fontScale,
                fontColor,
                thickness=thickness,
                lineType=lineType,
            )
    return output


def concat_n_videos(
    videos: dict, frames: int, gap: int = 0, text: bool = True, vid_names: dict = None,
) -> list:
    """
    Combines N color videos from a list of video paths.
    :param videos: List of video paths for concatenating
    :param frames: Number of frames to concatenate
    :param gap: Gap between images on a frame
    :param text: To overlay text
    :param vid_names: Text to overlay on each video
    :return: Video as list of concatenated frames
    """
    output = list()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.05
    fontColor = (255, 255, 255)
    thickness = 4
    lineType = 2
    text_height = 30
    num_vids = len(videos)
    # Position of text (x, y) -> origin is top left of picture with x increasing l->r and y increasing t->b
    # Position is bottom of leftmost character in string to overlay
    text_pos = (0, 0)
    if vid_names is None:
        vid_names = {
            "compressed": "Encoded",
            "original": "Original",
            "trained": "Post-Processed",
        }
    for i in range(frames):
        concat_frame = None
        for j, (vid_name, vid) in enumerate(videos.items()):
            temp = vid[i]
            # Add upper and lower borders
            h, w, c = temp.shape
            new_h = 3 * (2 * gap) + h + text_height
            new_frame = np.zeros(shape=(new_h, w, c))
            new_frame[4 * gap + text_height : -2 * gap, :] = temp
            if j % num_vids == 0:
                concat_frame = new_frame
                text_pos_x = int(0.5 * gap)
                text_pos_y = int(2 * gap + text_height)
                text_pos = (text_pos_x, text_pos_y)
            else:
                concat_frame = concat_images_horizontal(
                    concat_frame, new_frame, gap=gap, video=True
                )
                text_pos = (text_pos[0] + vid[i].shape[1] + gap, text_pos[1])
            frame_name = vid_names[vid_name]
            if text:
                cv2.putText(
                    concat_frame,
                    frame_name,
                    text_pos,
                    font,
                    fontScale,
                    fontColor,
                    thickness=thickness,
                    lineType=lineType,
                )
        output.append(concat_frame)
    return output


def get_image_corner(img, corner: str = "bottom_left", gap: int = 0) -> tuple:
    """
    Using an image and a corner, return the co-ordinates to write text in that corner
    :param img: image
    :param corner: desired corner to write in
    :param gap: gap between borders of image and text
    :return: co-ordinates of desired corner in image
    """
    if corner.lower() == "top_left":
        co_ords = (gap, gap + 30)
    elif corner.lower() == "top_right":
        co_ords = (img.shape[1] - gap, gap)
    elif corner.lower() == "bottom_right":
        co_ords = (img.shape[1] - gap, img.shape[0] - gap)
    else:
        # Default of "bottom_left"
        co_ords = (gap, img.shape[0] - gap)

    return co_ords


def unique_file(dest_path: str, directory=None) -> str:
    """
    Iterative increase the number on a file name to generate a unique file name
    :param dest_path: Original file name, which may already exist
    :param directory: Directory structure to create unique name in
    :return: Unique file name with appended index, for uniqueness
    """
    r_dir = os.getcwd()
    if directory is not None:
        os.chdir(directory)

    index = ""
    # Check if the folder name already exists
    while os.path.exists(index + dest_path):
        if index:
            index = "{}_".format(str(int(index[0:-1]) + 1))
        else:
            index = "1_"

    ret_val = index + dest_path
    if directory is not None:
        ret_val = os.path.join(directory, ret_val)
    os.chdir(r_dir)

    return ret_val


def create_empty_dir(dirname: str):
    """
    Ensure directory created is empty, delete if one exists already
    :param dirname: Directory to create
    :return:
    """
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)


def delete_folder(folder: str, up: bool = False):
    """
    Delete a folder's contents
    :param folder: folder to delete
    :param up: Delete folder and contents
    :return:
    """
    if up:
        folder = os.path.dirname(folder)
    shutil.rmtree(folder)
