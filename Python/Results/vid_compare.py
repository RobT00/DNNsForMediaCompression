"""
File to create side by side video for comparisons
Author: Robert Trew
"""
import os
import code_util
import results_util
import argparse
import numpy as np


def main(
    input_dir: str, output_dir: str, extension: str = "mkv", resolution: str = None
):
    """
    Create single video with side-by-side frames of input videos
    :param input_dir: Directory of videos to play side-by-side
    :param output_dir: Directory to output created video to
    :param extension: Extension to load videos from and save new video using
    :param resolution: Resolution of input videos
    :return:
    """
    colour_space = "BGR"  # for OpenCV
    video = True
    out_filename = f"combined_{resolution}" if resolution is not None else "combined"
    data_man = code_util.DataManagement(
        None, video, None, None, output_dir, None, colour_space
    )  # Create Basic instance
    videos = ["original", "compressed", "trained"]
    vid_name_dict = {
        "original": "Original",
        "compressed": "Encoded",
        "trained": "Post-Processed",
    }
    # load videos
    loaded_vids = dict()
    vid_metadata = dict()
    for vid in videos:
        vid_file = (
            f"{vid}_{resolution}.{extension}"
            if resolution is not None
            else f"{vid}.{extension}"
        )
        l_vid = data_man.load_video(os.path.join(input_dir, vid_file))
        vid_metadata[vid] = data_man.video_metadata(l_vid)
        loaded_vids[vid] = data_man.preprocess_video(
            l_vid, do_conversion=False, get_frames=range(vid_metadata[vid]["frames"])
        )
    l_vid.release()
    del l_vid

    framerate = min([j["fps"] for k, j in vid_metadata.items()])
    frames = vid_metadata["trained"]["frames"]
    org_frames = vid_metadata["original"]["frames"]
    comp_frames = vid_metadata["compressed"]["frames"]
    if frames <= org_frames:
        diff = int((org_frames - frames) / 2)
        if diff != 0:
            loaded_vids["original"] = loaded_vids["original"][diff:-diff]
    else:
        raise RuntimeError(f"Expected frames >= {frames}, got {org_frames}")
    if frames <= comp_frames:
        diff = int((comp_frames - frames) / 2)
        if diff != 0:
            loaded_vids["compressed"] = loaded_vids["compressed"][diff:-diff]
    else:
        raise RuntimeError(f"Expected frames >= {frames}, got {comp_frames}")
    # Using video 352x288
    clip_gap = 10
    concat_vid = np.asarray(
        results_util.concat_n_videos(
            loaded_vids, frames, gap=clip_gap, vid_names=vid_name_dict
        )
    )
    data_man.fps = float(framerate)
    print(concat_vid.shape)
    data_man.deprocess_video(
        concat_vid,
        os.path.join(output_dir, out_filename),
        file_format=extension,
        do_conversion=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Input video directory",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Output video directory",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-e",
        "--extension",
        dest="extension",
        help="Extension for saved videos",
        default="mkv",
        type=str,
    )

    parser.add_argument(
        "-r",
        "--resolution",
        dest="resolution",
        help="Specify resolution of videos to combine",
        default=None,
        type=str,
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir

    main(
        args.input_dir,
        args.output_dir,
        extension=args.extension,
        resolution=args.resolution,
    )
