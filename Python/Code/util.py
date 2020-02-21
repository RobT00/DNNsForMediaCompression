"""
File containing utility functions
"""
import os
import sys
import models
from datetime import timedelta
from keras.models import load_model, Model
from keras_preprocessing.image import (
    load_img,
    img_to_array,
    save_img,
    ImageDataGenerator,
)
from keras_applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from timeit import default_timer as timer  # Measured in seconds


class DataManagement:
    def __init__(self, script_dir, sequences, c_images, o_images, o_dir, precision):
        self.compare_dict = dict()
        self.input_dims = dict()
        self.script_dir = script_dir
        self.sequences = sequences
        self.compressed_data_path = c_images
        self.original_data_path = o_images
        self.out_path = o_dir
        # Temp variable for testing sequences - video
        self.frames = 5
        # self.frames = 250
        self.fps = None
        # self.train_datagen = ImageDataGenerator(
        #     rescale=None, dtype=precision, brightness_range=(0.1, 0.9)
        # )
        # self.test_datagen = ImageDataGenerator(rescale=None, dtype=precision)

    def preprocess_image(
        self,
        image_path,
        precision,
        data_format="channels_first",
        mode="div",
        plot=False,
        **dims,
    ):
        """
        Preprocess images, scale and convert to numpy array for feeding to model.
        :param image_path: Path to image
        :param data_format: Format to process image
        :param mode: Mode to process image
        :param precision: Precision for ndarray
        :param plot: Boolean - to plot image
        :param dims: Image dimensions to be used, tuple - (height, width, channels)
        :return: Processed image
        """
        img = load_img(image_path)
        img = img_to_array(img, dtype=precision)
        img = self.check_dims(img, dims.get("dims", img.shape))
        if mode == "div":
            img /= 255.0
        else:
            img = preprocess_input(img, data_format=data_format, mode=mode)
        if plot:
            plt.figure()
            plt.imshow(img.astype(float))
            plt.show()

        return img

    def preprocess_video(
        self,
        video_path,
        precision,
        data_format="channels_first",
        mode="div",
        plot=False,
        get_frames=None,
        **dims,
    ):
        """
        Preprocess images, scale and convert to numpy array for feeding to model.
        :param video_path: Path to video
        :param data_format: Format to process image
        :param mode: Mode to process image
        :param precision: Precision for ndarray
        :param plot: Boolean - to plot image
        :param get_frames: Frames to get -- when using generator
        :param dims: Image dimensions to be used, tuple - (height, width, channels)
        :return: Processed image
        """
        if get_frames is not None:
            # Video is already opened and streaming
            cap = video_path
        else:
            cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise UserWarning("Cannot read video, is codec installed?")
        # from tqdm import tqdm

        # while cap.isOpened():
        # Go to end of video
        # cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        # # Get duration in milliseconds
        # msec_dur = cap.get(cv2.CAP_PROP_POS_MSEC)
        # num_frames_1 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # fps = (num_frames_1 / msec_dur) * 1000
        # if not self.fps:
        #     self.fps = fps
        # num_frames_2 = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # fps_2 = (num_frames_2 / msec_dur) * 1000
        # frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # channels = cap.get(
        #     cv2.CAP_PROP_CHANNEL
        # )  # Doesn't seem to do much (with greyscale)
        # Return to start
        # cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
        vid = list()
        # TODO - Use all frames
        num_frames_1 = get_frames if get_frames is not None else range(int(self.frames))
        # for i in tqdm(num_frames_1, position=0, leave=True):
        for i in num_frames_1:
            cap.set(1, i)
            ret, frame = cap.read()
            # cv2.imshow('frame', frame)
            if not ret:
                break
            # frame = img_to_array(frame, dtype=precision)  # Needed to infer precision
            frame = frame.astype(precision, copy=False)
            frame = self.check_dims(frame, dims.get("dims", frame.shape))
            if mode == "div":
                frame /= 255.0
            else:
                frame = preprocess_input(frame, data_format=data_format, mode=mode)
            if plot:
                plt.figure()
                plt.imshow(frame.astype(float))
                plt.show()
            vid.append(frame)
        cap.release()

        return vid

    @staticmethod
    def load_video(video_path):
        """
        Function for loading a video to the stream, be sure to release the video when done!
        :param video_path: Path to video file
        :return: Video stream
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise UserWarning("Cannot read video, is codec installed?")

        return video

    def video_metadata(self, video):
        """
        Function for gather metadata about video, i.e length and fps
        :param video: Input video
        :return:
        """
        metadata = dict()
        video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        # Get duration in milliseconds
        msec_dur = video.get(cv2.CAP_PROP_POS_MSEC)
        metadata.update({"duration": msec_dur})
        num_frames = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        metadata.update({"frames": num_frames})
        fps = (num_frames / msec_dur) * 1000
        metadata.update({"fps": fps})
        if not self.fps:
            self.fps = fps
        return metadata

    @staticmethod
    def check_dims(image, desired_dims):
        """
        Check if the image dimensions are correct, raise error if not.
        :param image: Tuple of dimensions for image being processed - (height, width, channels)
        :param desired_dims: Tuple of desired image dimensions - (height, width, channels)
        :return: Image with correct dimensions, or raise error
        """

        def check_ok(image_dims, wanted_dims):
            """
            Check if the image dimensions are the same as the desired dimensions.
            :param image_dims: Tuple of image dimensions - (height, width, channels)
            :param wanted_dims: Tuple of desired image dimensions - (height, width, channels)
            :return: True / False
            """
            return image_dims == wanted_dims

        # TODO - Crop images, resize
        # i_height, i_width, i_channels = image.shape
        if not check_ok(image.shape, desired_dims):
            # if width != i_width:
            #     if height == i_width:
            #         # image = image.reshape([i_width, i_height, i_channels])
            #         image = np.rot90(image)
            #         i_width, i_height, i_channels = image.shape
            # if i_channels != channels:
            #     image = image.reshape([i_height, i_width, channels])
            if image.shape[0] != desired_dims[0]:
                if image.shape[1] == desired_dims[0]:
                    image = np.rot90(image)
            if image.shape[2] != desired_dims[2]:
                image = image.reshape([image.shape[0], image.shape[1], desired_dims[2]])
            if not check_ok(image.shape, desired_dims):
                raise ValueError(
                    "Image: {} doesn't match desired dimensions: {}".format(
                        image.shape, desired_dims
                    )
                )

        return image

    @staticmethod
    def deprocess_image(img, data_format="channels_first", plot=False):
        """
        Convert the predicted image from the model back to [0..255] for viewing / saving.
        :param img: Predicted image
        :param data_format: Format to process image
        :param plot: Boolean - to plot image for viewing
        :return: Restored image
        """
        # if data_format == "channels_first":
        #     img = img.reshape((3, img.shape[1], img.shape[2]))
        #     img = img.transpose((1, 2, 0))
        # else:
        #     img = img.reshape((img.shape[1], img.shape[2], 3))
        # img /= 2.0
        # img += 0.
        if img.shape[-1] != 3:
            raise ValueError("Not RGB")
        if len(img.shape) == 4:
            img = img.reshape((img.shape[1], img.shape[2], img.shape[3]))
        try:
            img *= 255.0
        except RuntimeWarning:
            pass
        img = np.clip(img, 0, 255).astype("uint8")

        if plot:
            plt.figure()
            plt.imshow(img)
            plt.show()

        return img

    def get_training_data(self, **kwargs):
        if self.sequences:
            return self.get_training_videos(**kwargs)
        else:
            return self.get_training_images(**kwargs)

    def get_training_videos(self, precision="float32", img_format="mp4", plot=False):
        # raise UserWarning("The generator must be used when training on video")
        compressed_video = dict()
        for compression_level in range(1):
            for filename in glob.glob(self.compressed_data_path + f"/*.{img_format}"):
                vid = self.preprocess_video(
                    filename, precision, mode="div", plot=plot, **self.input_dims
                )
                if not self.input_dims:
                    self.input_dims.update({"dims": vid[0].shape})
                compressed_video.setdefault(compression_level, list()).append(vid)

        compressed_video_array = list()
        for i in compressed_video.values():
            compressed_video_array.extend(i)

        compressed_video_array = np.asarray(compressed_video_array, dtype=precision)

        return compressed_video, compressed_video_array

    def get_training_images(self, precision="float32", img_format="jpg", plot=False):
        compressed_images = dict()
        for compression_level in os.listdir(self.compressed_data_path):
            for filename in glob.glob(
                # fmt: off
                os.path.join(self.compressed_data_path,
                             compression_level) + f"/*.{img_format}"
                # fmt: on
            ):
                img = self.preprocess_image(
                    filename, precision, mode="div", plot=plot, **self.input_dims
                )
                if not self.input_dims:
                    self.input_dims.update({"dims": img.shape})
                compressed_images.setdefault(compression_level, list()).append(img)

        compressed_images_array = list()
        for i in compressed_images.values():
            compressed_images_array.extend(i)

        compressed_images_array = np.asarray(compressed_images_array, dtype=precision)

        return compressed_images, compressed_images_array

    def get_labels(self, num_training, **kwargs):
        if self.sequences:
            return self.get_label_videos(num_training, **kwargs)
        else:
            return self.get_label_images(num_training, **kwargs)

    def get_label_videos(
        self, num_compressed_videos, precision="float32", img_format="y4m", plot=False
    ):
        # raise UserWarning("The generator must be used when training on video")
        original_videos = list()
        for filename in glob.glob(self.original_data_path + f"/*.{img_format}"):
            vid = self.preprocess_video(
                filename, precision, mode="div", plot=plot, **self.input_dims
            )
            if not self.input_dims:
                self.input_dims.update({"dims": vid[0].shape})
            original_videos.append(vid)

        n = num_compressed_videos // len(original_videos)
        original_videos *= n
        original_videos = np.asarray(original_videos, dtype=precision)

        return original_videos

    def get_label_images(
        self, num_compressed_images, precision="float32", img_format="png", plot=False
    ):
        original_images = list()
        for filename in glob.glob(self.original_data_path + f"/*.{img_format}"):
            img = self.preprocess_image(
                filename, precision, mode="div", plot=plot, **self.input_dims
            )
            if not self.input_dims:
                self.input_dims.update({"dims": img.shape})
            original_images.append(img)

        n = num_compressed_images // len(original_images)
        original_images *= n
        original_images = np.asarray(original_images, dtype=precision)

        return original_images

    def get_input_dims(self):
        # TODO - Open an image/video at random, get metadata, but shape so that
        # width > height
        if self.sequences:
            # Add number of frames
            # d = self.input_dims.get("dims", (144, 176, 3))
            d = self.input_dims.get("dims", (288, 352, 3))
            d = (self.frames,) + d  # Frames first
            # d = (None,) + d  # Unspecified number of frames
            # d += (300,)  # Frames last
            # self.input_dims.update({"dims": d})
        else:
            d = self.input_dims.get("dims", (512, 768, 3))
        return d

    def generator_function(self):
        files = [
            os.path.join(self.compressed_data_path, f)
            for f in os.listdir(self.compressed_data_path)
        ]
        # if not self.input_dims:
        if self.sequences:
            function = self.video_generator
        else:
            function = self.image_generator
        return files, function

    def image_generator(
        self, files, batch_size=42, precision="float32", file_type="png"
    ):
        # Get input image
        # input_img = self.preprocess_image()
        # Match output image
        # out_name = input_img.name.split("_")[0]
        # Do glob to find matching image
        # for filename in glob.glob(self.compressed_images_path + f"/*.{img_format}"):
        #     img = self.preprocess_image(
        #         filename, precision, mode="div", plot=plot, **self.input_dims
        #     )
        #     if not self.input_dims:
        #         self.input_dims.update({"dims": img.shape})
        #     compression_level = int(
        #         os.path.basename(filename).split(img_format)[0].split("_")[1][:-1]
        #     )
        #     compressed_images.setdefault(compression_level, list()).append(img)
        # out_img = self.preprocess_image(glob.find.image(out_name))
        # Return pairwise training set
        # return (input_img, out_img)
        self.input_dims.update({"dims": self.get_input_dims()})
        while True:
            # Select files for the batch
            batch_paths = np.random.choice(a=files, size=batch_size)
            batch_input = list()
            batch_output = list()

            # Read in each input, perform pre-processing and get labels (original images)
            for input_img in batch_paths:
                input = self.preprocess_image(input_img, precision, **self.input_dims)

                file_name = os.path.basename(input_img).split("_")[0]
                file_path = os.path.join(
                    self.original_data_path, f"{file_name}.{file_type}"
                )
                output = self.preprocess_image(file_path, precision, **self.input_dims)

                batch_input.append(input)
                batch_output.append(output)

            # Return a tuple of (input, output) to feed network
            batch_x = np.asarray(batch_input, dtype=precision)
            batch_y = np.asarray(batch_output, dtype=precision)

            # return batch_x, batch_y
            yield batch_x, batch_y

    def video_generator(
        self, files, batch_size=4, precision="float32", file_type="y4m"
    ):
        while True:
            # Select files for the batch
            batch_paths = np.random.choice(a=files, size=batch_size)
            batch_input = list()
            batch_output = list()

            # Read in each input, perform pre-processing and get labels (original images)
            for input_video in batch_paths:
                # Load video
                cap = self.load_video(input_video)
                # cap = cv2.VideoCapture(input_video)
                # if not cap.isOpened():
                #     raise UserWarning("Cannot read video, is codec installed?")
                # Randomly gather self.frames consecutive frames
                metadata = self.video_metadata(cap)
                # TODO - Handle blank before / after frames
                start_frame = int(
                    np.random.choice(a=metadata.get("frames") - self.frames, size=1)
                )
                frames = np.arange(start_frame, start_frame + self.frames)
                # for i in frames:
                #     ret, frame = cap.get(i)
                #     if not ret:
                #         raise UserWarning(f"Frame {i} not found!")
                input = self.preprocess_video(
                    cap, precision, get_frames=frames, **self.input_dims
                )
                cap.release()
                # Repeat for output
                file_name = os.path.basename(input_video).split(f".mp4")[0]
                file_path = os.path.join(
                    self.original_data_path, f"{file_name}.{file_type}"
                )

                cap = self.load_video(file_path)
                output = self.preprocess_video(
                    cap, precision, get_frames=frames, **self.input_dims
                )
                cap.release()

                batch_input.append(input)
                batch_output.append(output)

            # Return a tuple of (input, output) to feed network
            batch_x = np.asarray(batch_input, dtype=precision)
            batch_y = np.asarray(batch_output, dtype=precision)

            # return batch_x, batch_y
            yield batch_x, batch_y

    @staticmethod
    def unique_file(dest_path):
        """
        Iterative increase the number on a file name to generate a unique file name
        :param dest_path: Original file name, which may already exist
        :return: Unique file name with appended index, for uniqueness
        """
        index = ""
        # Check if the folder name already exists
        while os.path.exists(index + dest_path):
            if index:
                index = "({}) ".format(str(int(index[1:-2]) + 1))
            else:
                index = "(1) "

        return index + dest_path

    def output_results(self, model, input_data, labels, **kwargs):
        if self.sequences:
            return self.output_results_videos(model, input_data, labels, **kwargs)
        else:
            return self.output_results_images(model, input_data, labels, **kwargs)

    def output_results_images(
        self,
        model,
        input_images,
        labels,
        training_data=None,
        precision="float32",
        loss_fn="MS-SSIM",
        validation=True,
    ):
        f_name = ""
        return_dir = os.getcwd()
        self.out_path = os.path.join(self.out_path, model.name)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        os.chdir(self.out_path)
        if training_data:
            # Create folder name based on params
            f_name += "optimiser={} epochs={} batch_size={}".format(
                training_data.model.optimizer.iterations.name.split("/")[0],
                # training_data.params["epochs"],
                len(training_data.epoch),
                training_data.params["batch_size"],
            )

            # Create plots to save training records
            fig_1 = plt.figure()
            plt.plot(
                np.asarray(training_data.history["loss"]) * -1.0,
                label=f"{loss_fn} Training Loss",
            )
            if validation:
                plt.plot(
                    np.asarray(training_data.history["val_loss"]) * -1.0,
                    label=f"{loss_fn} Validation Loss",
                )
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.legend()
            plt.title(f_name)
            # plt.show()

            fig_2 = plt.figure()
            plt.plot(
                np.asarray(training_data.history["tf_psnr"]) * -1.0,
                label="PSNR Training Loss",
            )
            if validation:
                plt.plot(
                    np.asarray(training_data.history["val_tf_psnr"]) * -1.0,
                    label="PSNR Validation Loss",
                )
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.legend()
            plt.title(f_name)

            f_name += " metrics={} model={} precision={}".format(
                ",".join(
                    [
                        i
                        for i in training_data.params["metrics"]
                        if i != "loss" and i[:4] != "val_"
                    ]
                ),
                model.name,
                precision,
            )

            out_path = os.path.join(self.out_path, self.unique_file(f_name))

            # Save generated plots
            p_dir = os.path.join(out_path, "Plots")
            os.makedirs(p_dir)
            os.chdir(p_dir)

            fig_1.savefig(f"{loss_fn}.png")
            fig_2.savefig("PSNR.png")

            # Save model
            m_dir = os.path.join(out_path, "Model")
            os.makedirs(m_dir)
            os.chdir(m_dir)

            model.save("{}.h5".format(model.name))
        else:
            f_name += "loaded_model={} precision={}".format(model.name, precision)
            out_path = os.path.join(self.out_path, self.unique_file(f_name))

        # Save sample training and validation images
        t_dir = os.path.join(out_path, "Training")
        output_time = 0.0
        i = 0  # So that i will always be defined
        num_images = 0
        if type(input_images) is dict:
            for compression_level, images in input_images.items():
                for i, (t_im, o_im) in enumerate(
                    # TODO - stop black adding whitespace before ':'
                    # fmt: off
                    zip(images, labels[num_images: len(images) + num_images]), start=1
                    # fmt: on
                ):
                    output_time += self.output_helper_images(
                        t_dir,
                        t_im,
                        o_im,
                        model,
                        output_append=[str(compression_level), str(i)],
                    )
                num_images += i
        else:
            for i, (t_im, o_im) in enumerate(zip(input_images, labels), start=1):
                output_time += self.output_helper_images(
                    t_dir, t_im, o_im, model, output_append=str(i)
                )
            num_images += i

        os.chdir(return_dir)

        return timedelta(milliseconds=output_time / num_images)

    def output_helper_images(
        self,
        output_directory,
        input_image,
        original_image,
        model,
        output_append=None,
        plot=False,
    ):
        if type(output_append) is list:
            for appendage in output_append:
                output_directory = os.path.join(output_directory, appendage)
        elif output_append:
            output_directory = os.path.join(output_directory, output_append)

        os.makedirs(output_directory)
        os.chdir(output_directory)

        train_im = np.expand_dims(input_image, axis=0)
        start = timer()
        train_pred = model.predict(train_im)
        end = timer()
        save_img(
            "original.png",
            self.deprocess_image(np.expand_dims(original_image, axis=0), plot=plot),
        )
        save_img("compressed.png", self.deprocess_image(train_im, plot=plot))
        save_img("trained.png", self.deprocess_image(train_pred, plot=plot))

        return (end - start) * 1000

    def output_results_videos(
        self,
        model,
        input_videos,
        labels,
        training_data=None,
        precision="float32",
        loss_fn="mse",
        validation=True,
    ):
        f_name = ""
        return_dir = os.getcwd()
        self.out_path = os.path.join(self.out_path, model.name)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        os.chdir(self.out_path)
        if training_data:
            # Create folder name based on params
            # TODO - Mono or Colour
            f_name += "optimiser={} epochs={} batch_size={}".format(
                training_data.model.optimizer.iterations.name.split("/")[0],
                # training_data.params["epochs"],
                len(training_data.epoch),
                training_data.params["batch_size"],
            )

            # Create plots to save training records
            fig_1 = plt.figure()
            plt.plot(
                np.asarray(training_data.history["loss"]) * -1.0,
                label=f"{loss_fn} Training Loss",
            )
            if validation:
                plt.plot(
                    np.asarray(training_data.history["val_loss"]) * -1.0,
                    label=f"{loss_fn} Validation Loss",
                )
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.legend()
            plt.title(f_name)
            # plt.show()

            # fig_2 = plt.figure()
            # plt.plot(
            #     np.asarray(training_data.history["tf_psnr"]) * -1.0,
            #     label="PSNR Training Loss",
            # )
            # if validation:
            #     plt.plot(
            #         np.asarray(training_data.history["val_tf_psnr"]) * -1.0,
            #         label="PSNR Validation Loss",
            #     )
            # plt.xlabel("Epochs")
            # plt.ylabel("Score")
            # plt.legend()
            # plt.title(f_name)

            f_name += " metrics={} model={} precision={}".format(
                ",".join(
                    [
                        i
                        for i in training_data.params["metrics"]
                        if i != "loss" and i[:4] != "val_"
                    ]
                ),
                model.name,
                precision,
            )

            out_path = os.path.join(self.out_path, self.unique_file(f_name))

            # Save generated plots
            p_dir = os.path.join(out_path, "Plots")
            os.makedirs(p_dir)
            os.chdir(p_dir)

            fig_1.savefig(f"{loss_fn}.png")
            # fig_2.savefig("PSNR.png")

            # Save model
            m_dir = os.path.join(out_path, "Model")
            os.makedirs(m_dir)
            os.chdir(m_dir)

            model.save("{}.h5".format(model.name))
        else:
            f_name += "loaded_model={} precision={}".format(model.name, precision)
            out_path = os.path.join(self.out_path, self.unique_file(f_name))

        # Save sample training and validation images
        t_dir = os.path.join(out_path, "Training")
        output_time = 0.0
        i = 0  # So that i will always be defined
        num_videos = 0
        if type(input_videos) is dict:
            for compression_level, videos in input_videos.items():
                for i, (t_im, o_im) in enumerate(
                    # TODO - stop black adding whitespace before ':'
                    # fmt: off
                    zip(videos, labels[num_videos: len(videos) + num_videos]), start=1
                    # fmt: on
                ):
                    output_time += self.output_helper_video(
                        t_dir,
                        t_im,
                        o_im,
                        model,
                        output_append=[str(compression_level), str(i)],
                    )
                num_videos += i
        else:
            for i, (t_im, o_im) in enumerate(zip(input_videos, labels), start=1):
                output_time += self.output_helper_video(
                    t_dir, t_im, o_im, model, output_append=str(i)
                )
            num_videos += i

        os.chdir(return_dir)

        return timedelta(milliseconds=output_time / num_videos)

    def output_helper_video(
        self,
        output_directory,
        input_video,
        original_video,
        model,
        output_append=None,
        plot=False,
    ):
        if type(output_append) is list:
            for appendage in output_append:
                output_directory = os.path.join(output_directory, appendage)
        elif output_append:
            output_directory = os.path.join(output_directory, output_append)

        os.makedirs(output_directory)
        os.chdir(output_directory)

        train_video = np.expand_dims(input_video, axis=0)
        start = timer()
        train_pred = model.predict(train_video)
        end = timer()
        # save_img(
        #     "original.mp4",
        #     self.deprocess_image(np.expand_dims(original_video, axis=0), plot=plot),
        # )
        self.deprocess_video(original_video, "original")
        # save_img("compressed.mp4", self.deprocess_image(train_im, plot=plot))
        self.deprocess_video(train_video.squeeze(axis=0), "compressed")
        # save_img("trained.mp4", self.deprocess_image(train_pred, plot=plot))
        self.deprocess_video(train_pred.squeeze(axis=0), "trained")

        return (end - start) * 1000

    def deprocess_video(self, video, file_name, file_format="avi", **kwargs):
        """
        Convert the video from a numpy array back to [0..255] for viewing / saving.
        :param video: Video as numpy array
        :param file_name: Name to save video under
        :param file_format: Format / container to save video as
        :param fps: Playback speed of video
        :return: Restored image
        """
        file_name = file_name + "." + file_format
        width = video.shape[-2]
        height = video.shape[-3]

        # initialise video writer
        # https://stackoverflow.com/questions/51914683/how-to-make-video-from-an-updating-numpy-array-in-python
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        writer = cv2.VideoWriter(file_name, fourcc, self.fps, (width, height))

        for frame in video:
            frame = self.deprocess_image(frame, **kwargs)
            writer.write(frame)
        # Close video writer
        writer.release()

    @staticmethod
    def get_model_from_string(classname):
        return getattr(sys.modules[__name__].models, classname)

    @staticmethod
    def load_model_from_path(model_path):
        return load_model(model_path, compile=False)

    @staticmethod
    def loaded_model(model):
        return type(model) == Model
