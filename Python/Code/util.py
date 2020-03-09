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
from scipy import ndimage
import glob
import cv2
import re
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

    def preprocess_image(self, image_path, precision, mode=None, plot=False, **dims):
        """
        Preprocess images, scale and convert to numpy array for feeding to model.
        :param image_path: Path to image
        :param mode: Mode to process image
        :param precision: Precision for ndarray
        :param plot: Boolean - to plot image
        :param dims: Image dimensions to be used, tuple - (height, width, channels)
        :return: Processed image
        """
        img = load_img(image_path)
        img = img_to_array(img, dtype=precision)
        img = self.check_dims(img, dims.get("dims", img.shape))
        # if mode == "div":
        #     img /= 255.0
        # else:
        #     img = preprocess_input(img, data_format=data_format, mode=mode)
        img = self.do_augmentation(mode, img)
        if plot:
            plt.figure()
            plt.imshow(img.astype(float))
            plt.show()

        return img

    def preprocess_video(
        self, video_path, precision, mode=None, plot=False, get_frames=None, **dims
    ):
        """
        Preprocess images, scale and convert to numpy array for feeding to model.
        :param video_path: Path to video
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
        frames = get_frames if get_frames is not None else range(int(self.frames))
        # for i in tqdm(num_frames_1, position=0, leave=True):
        for i in frames:
            cap.set(1, i)
            ret, frame = cap.read()
            # cv2.imshow('frame', frame)
            if not ret:
                break
            # frame = img_to_array(frame, dtype=precision)  # Needed to infer precision
            # TODO - Does RGB help ?
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert correct colour channels
            # TODO Does YUV help ?
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)  # Convert to YUV
            frame = frame.astype(precision, copy=False)
            frame = self.check_dims(frame, dims.get("dims", frame.shape))
            # if mode == "div":
            #     frame /= 255.0
            # else:
            #     frame = preprocess_input(frame, data_format=data_format, mode=mode)
            frame = self.do_augmentation(mode, frame)
            if plot:
                plt.figure()
                plt.imshow(frame.astype(float))
                plt.show()
            vid.append(frame)
        cap.release()

        return vid

    @staticmethod
    def do_augmentation(aug_type: dict, img):
        """
        Function to do some augmentation to an image, to expand training
        :param img: Input image
        :param aug_type: Type of augmentation to perform, may be multiple
        :return: Augmented image
        """
        precision = img.dtype

        def add_noise(noise_typ, image):
            # noisy = np.copy(image)
            if noise_typ == "gaussian":
                row, col, ch = image.shape
                mean = 0
                var = 0.1
                sigma = var ** 0.5
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                gauss = gauss.reshape(row, col, ch)
                noisy = image + gauss
            elif noise_typ == "s&p":
                row, col, ch = image.shape
                s_vs_p = 0.5
                amount = 0.004
                # Salt mode
                num_salt = np.ceil(amount * image.size * s_vs_p)
                coords = [
                    np.random.randint(0, i - 1, int(num_salt)) for i in image.shape
                ]
                # image[coords] = 255.0
                image[tuple(coords)] = 255.0

                # Pepper mode
                num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
                coords = [
                    np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape
                ]
                # image[coords] = 0.0
                image[tuple(coords)] = 0.0
                noisy = image
            elif noise_typ == "poisson":
                vals = len(np.unique(image))
                vals = 2 ** np.ceil(np.log2(vals))
                noisy = np.random.poisson(image * vals) / float(vals)
            elif noise_typ == "speckle":
                row, col, ch = image.shape
                gauss = np.random.randn(row, col, ch)
                gauss = gauss.reshape(row, col, ch)
                noisy = image + image * gauss
            else:
                noisy = image
            return noisy

        def rotate(degrees, image):
            return ndimage.rotate(image, degrees, reshape=False)

        def brightness(level, image):
            return cv2.convertScaleAbs(image, alpha=level, beta=0)

        def contrast(level, image):
            return cv2.convertScaleAbs(image, alpha=1.0, beta=level)

        if aug_type is not None:
            for aug_name, aug in aug_type.items():
                if "noise" in aug_name.lower():
                    img = add_noise(aug, img)
                elif "rotate" in aug_name.lower():
                    img = rotate(aug, img)
                elif "brightness" in aug_name.lower():
                    img = brightness(aug, img)
                elif "contrast" in aug_name.lower():
                    img = contrast(aug, img)
                # Ensure image limits and dtype after each pass
                img = np.clip(img, 0.0, 255.0).astype(dtype=precision)
        img /= 255.0

        return img

    @staticmethod
    def get_augemntations() -> dict:
        """
        Using a random number generator, the augmentation(s) to be used are returned as a dictionary
        :return: Dictionary containing augmentations to apply
        """
        noise_types = ["gaussian", "s&p", "poisson", "speckle"]
        rotations = [
            1,
            5,
            10,
            15,
            20,
            30,
            40,
            45,
            55,
            60,
            75,
            90,
            120,
            150,
            160,
            180,
            270,
            290,
        ]
        brightness_values = [0.25, 0.5, 0.75, 0.85, 0.99, 1.01, 1.25, 1.5, 1.75, 2]
        contrast_values = [1, 5, 10, 20, 25, 33, 42, 50, 60, 75, 85, 99]
        augmentations = {
            "noise": noise_types,
            "rotate": rotations,
            "brightness": brightness_values,
            "contrast": contrast_values,
        }

        chosen_augmentations = dict()
        num_range = 10000
        num = np.random.choice(a=num_range)

        if num < num_range / 2:  # 5000 / 5000
            pass
        else:  # 1250 / 1250 / 1250 / 1250
            num -= num_range / 2
            num_range /= 2
            if num < num_range / 4:
                num = 1
            elif num < num_range / 2:
                num = 2
            elif num < num_range * 0.75:
                num = 3
            else:
                num = 4
            keys = np.random.choice(
                a=list(augmentations.keys()),
                replace=False,
                size=num,
                p=[0.265, 0.245, 0.245, 0.245],
            )

            for key in keys:
                chosen_augment = np.random.choice(a=augmentations[key])
                chosen_augmentations.update({key: chosen_augment})

        return chosen_augmentations

    @staticmethod
    def load_video(video_path):
        """
        Function for loading a video to the stream, be sure to release the video when done!
        :param video_path: Path to video file
        :return: Video stream
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise UserWarning(f"Cannot read video at {video_path}\nIs codec installed?")

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

        # i_height, i_width, i_channels = image.shape
        if not check_ok(image.shape, desired_dims):
            if image.shape[0] != desired_dims[0]:
                if image.shape[1] == desired_dims[0]:
                    image = np.rot90(image)
            if image.shape[2] != desired_dims[2]:
                image = image.reshape([image.shape[0], image.shape[1], desired_dims[2]])
            if not check_ok(image.shape, desired_dims):
                try:
                    image = cv2.resize(
                        image,
                        (desired_dims[0], desired_dims[1]),
                        interpolation=cv2.INTER_AREA,
                    )
                except ValueError:
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
                    filename, precision, plot=plot, **self.input_dims
                )
                if not self.input_dims:
                    self.set_input_dims(vid[0].shape)
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
                    filename, precision, plot=plot, **self.input_dims
                )
                if not self.input_dims:
                    self.set_input_dims(img.shape)
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
                filename, precision, plot=plot, **self.input_dims
            )
            if not self.input_dims:
                self.set_input_dims(vid[0].shape)
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
                filename, precision, plot=plot, **self.input_dims
            )
            if not self.input_dims:
                self.set_input_dims(img.shape)
            original_images.append(img)

        n = num_compressed_images // len(original_images)
        original_images *= n
        original_images = np.asarray(original_images, dtype=precision)

        return original_images

    def get_input_dims(self):
        # TODO - Use runtime arg ?
        # width > height
        if self.sequences:
            # Add number of frames
            # d = self.input_dims.get("dims", (144, 176, 3))
            # d = self.input_dims.get("dims", (288, 352, 3))
            # d = self.input_dims.get("dims", (48, 48, 3))
            d = self.input_dims.get("dims", (128, 128, 3))
            # d = self.input_dims.get("dims", (256, 256, 3))
            d = (self.frames,) + d  # Frames first
            # d = (None,) + d  # Unspecified number of frames
            # d += (300,)  # Frames last
            # self.input_dims.update({"dims": d})
        else:
            d = self.input_dims.get("dims", (512, 768, 3))
        return d

    def set_input_dims(self, input_dims: tuple):
        """
        Helper method to set the input dims
        :param input_dims: Tuple of dimensions for input to model
        :return: Updates class with input doms
        """
        self.input_dims.update({"dims": input_dims})

    def generator_function(self, validate=True, split=0.2):
        files = [
            os.path.join(self.compressed_data_path, f)
            for f in os.listdir(self.compressed_data_path)
        ]
        if validate:
            np.random.shuffle(files)
            split_size = int(len(files) * split)
            train_files = files[:-split_size]
            validate_files = files[-split_size:]
            # files = (train_files, validate_files)
        else:
            train_files = files
            validate_files = None
        if self.sequences:
            function = self.video_generator
        else:
            function = self.image_generator
        return train_files, validate_files, function

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
                augment = self.get_augemntations()
                input = self.preprocess_image(
                    input_img, precision, mode=augment, **self.input_dims
                )

                file_name = os.path.basename(input_img).split("_")[0]
                file_path = os.path.join(
                    self.original_data_path, f"{file_name}.{file_type}"
                )
                if "rotate" in augment:
                    augment = {"rotate": augment["rotate"]}
                else:
                    augment = None
                output = self.preprocess_image(
                    file_path, precision, mode=augment, **self.input_dims
                )

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
        dims = self.get_input_dims()
        self.set_input_dims(dims[1:])
        while True:
            # Select files for the batch
            batch_paths = np.random.choice(a=files, size=batch_size)
            batch_input = list()
            batch_output = list()

            # Read in each input, perform pre-processing and get labels (original images)
            for input_video in batch_paths:
                augment = self.get_augemntations()
                # Load video
                cap = self.load_video(input_video)
                # cap = cv2.VideoCapture(input_video)
                # if not cap.isOpened():
                #     raise UserWarning("Cannot read video, is codec installed?")
                # Randomly gather self.frames consecutive frames
                metadata = self.video_metadata(cap)
                # TODO - Handle blank before / after frames
                start_frame = np.random.choice(a=metadata.get("frames") - self.frames)
                frames = np.arange(start_frame, start_frame + self.frames)
                # for i in frames:
                #     ret, frame = cap.get(i)
                #     if not ret:
                #         raise UserWarning(f"Frame {i} not found!")
                input = self.preprocess_video(
                    cap, precision, get_frames=frames, mode=augment, **self.input_dims
                )
                cap.release()
                # Repeat for output
                re_exp = "(_\d+\.mp4$)"
                # base_file = os.path.basename(input_video)
                # file_name = base_file.strip(re_exp.findall(base_file)[0])
                file_name = self.get_base_filename(input_video, re_exp)
                # file_name = os.path.basename(input_video).split(f".mp4")[0]
                file_path = os.path.join(
                    self.original_data_path, f"{file_name}.{file_type}"
                )
                mid_frame = np.asarray([frames[len(frames) // 2]])
                cap = self.load_video(file_path)
                if "rotate" in augment:
                    augment = {"rotate": augment["rotate"]}
                else:
                    augment = None
                output = self.preprocess_video(
                    cap,
                    precision,
                    get_frames=mid_frame,
                    mode=augment,
                    **self.input_dims,
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
    def get_base_filename(full_name, file_regex):
        """
        Helper function to retrieve a base file name for matching training and label data
        :param full_name: Path to training file
        :param file_regex: Regex to remove from training file to get match
        :return:
        """
        re_exp = re.compile(file_regex)
        base_file = os.path.basename(full_name)
        base_filename = base_file.strip(re_exp.findall(base_file)[0])

        return base_filename

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

    # def output_results(self, model, input_data, labels, **kwargs):
    def output_results(self, model, *args, **kwargs):
        if self.sequences:
            return self.output_results_videos(model, **kwargs)
        else:
            return self.output_results_images(model, *args, **kwargs)

    def output_results_images(
        self,
        model,
        input_images,
        labels,
        training_data=None,
        precision="float32",
        loss_fn="MS-SSIM",
        **kwargs,
    ):
        f_name = ""
        return_dir = os.getcwd()
        if training_data:
            training_dims = f"{model.input_shape[2]}x{model.input_shape[1]}"
            self.out_path = os.path.join(self.out_path, model.name, training_dims)
            self.out_path = os.path.join(self.out_path, model.name)
            if not os.path.exists(self.out_path):
                os.makedirs(self.out_path)
            os.chdir(self.out_path)
            # Create folder name based on params
            f_name += "optimiser={}_epochs={}_batch_size={}_lr={}".format(
                training_data.model.optimizer.iterations.name.split("/")[0],
                # training_data.params["epochs"],
                len(training_data.epoch),
                training_data.params["batch_size"],
                training_data.params["lr"],
            )

            # Create plots to save training records
            fig_1 = plt.figure()
            plt.plot(
                np.asarray(training_data.history["loss"]) * -1.0,
                label=f"{loss_fn} Training Loss",
            )
            # if validation:
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
            # if validation:
            plt.plot(
                np.asarray(training_data.history["val_tf_psnr"]) * -1.0,
                label="PSNR Validation Loss",
            )
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.legend()
            plt.title(f_name)

            fig_3 = plt.figure()
            plt.plot(np.asarray(training_data.history["lr"]), label="Learning Rate")
            plt.xlabel("Epochs")
            plt.ylabel("Learning Rate")
            plt.legend()
            plt.title(f_name)

            f_name += "_metrics={}_model={}_precision={}".format(
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
            fig_3.savefig("lr.png")

            # Save model
            m_dir = os.path.join(out_path, "Model")

            self.do_model_saving(model, m_dir)
        else:
            # f_name += "loaded_model={}_precision={}".format(model.name, precision)
            f_name += "loaded_model"
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
        # input_videos,
        # labels,
        training_data=None,
        precision="float32",
        loss_fn="MS-SSIM",
        **kwargs,
    ):
        f_name = ""
        return_dir = os.getcwd()
        if training_data:
            training_dims = f"{model.input_shape[3]}x{model.input_shape[2]}"
            encoder = self.compressed_data_path.split(os.sep)[-1]
            self.out_path = os.path.join(self.out_path, model.name, encoder)
            if "LowQual" in self.compressed_data_path.split(os.sep):
                self.out_path = os.path.join(self.out_path, "LowQual")
            self.out_path = os.path.join(self.out_path, training_dims)
            if not os.path.exists(self.out_path):
                os.makedirs(self.out_path)
            os.chdir(self.out_path)
            # Create folder name based on params
            f_name += "optimiser={}_epochs={}_batch_size={}_lr={}".format(
                training_data.model.optimizer.iterations.name.split("/")[0],
                # training_data.params["epochs"],
                len(training_data.epoch),
                training_data.params["batch_size"],
                training_data.params["lr"],
            )

            # Create plots to save training records
            fig_1 = plt.figure()
            plt.plot(
                np.asarray(training_data.history["loss"]) * -1.0,
                label=f"{loss_fn} Training Loss",
            )
            # if validation:
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
                np.asarray(training_data.history["mean_squared_error"]),
                label="MSE Training Loss",
            )
            # if validation:
            plt.plot(
                np.asarray(training_data.history["val_mean_squared_error"]),
                label="MSE Validation Loss",
            )
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.legend()
            plt.title(f_name)

            fig_3 = plt.figure()
            plt.plot(np.asarray(training_data.history["lr"]), label="Learning Rate")
            plt.xlabel("Epochs")
            plt.ylabel("Learning Rate")
            plt.legend()
            plt.title(f_name)

            f_name += "_metrics={}_model={}_precision={}".format(
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
            fig_2.savefig("mse.png")
            fig_3.savefig("lr.png")

            # Save model
            m_dir = os.path.join(out_path, "Model")

            self.do_model_saving(model, m_dir)

        else:
            # f_name += "loaded_model={}_precision={}".format(model.name, precision)
            f_name += "loaded_model"
            out_path = os.path.join(self.out_path, self.unique_file(f_name))

        # Save sample training and validation images
        t_dir = os.path.join(out_path, "Training")
        output_time = 0.0
        i = 0  # So that i will always be defined
        num_videos = 0
        input_videos = os.listdir(self.compressed_data_path)
        re_exp = r"(_\d+\.mp4$)"
        qualities = sorted(
            set(
                [
                    # fmt: off
                    j[1: len(j) - len(".mp4")]
                    # fmt: on
                    for j in list(set([re.findall(re_exp, i)[0] for i in input_videos]))
                ]
            ),
            reverse=True,
        )
        qualities = qualities[:3]  # get top 3
        for quality in qualities:
            match_string = f"_{quality}.mp4"
            for i, video_file in enumerate(
                glob.glob(self.compressed_data_path + f"/*{match_string}"), start=1
            ):
                base_name = self.get_base_filename(video_file, re_exp)
                original_video = os.path.join(
                    self.original_data_path, f"{base_name}.y4m"
                )
                output_time += self.output_helper_video(
                    t_dir,
                    video_file,
                    original_video,
                    model,
                    output_append=[f"crf={quality}", base_name],
                    precision=precision,
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
        precision="float32",
    ):
        if type(output_append) is list:
            for appendage in output_append:
                output_directory = os.path.join(output_directory, appendage)
        elif output_append:
            output_directory = os.path.join(output_directory, output_append)

        os.makedirs(output_directory)
        os.chdir(output_directory)
        # TODO - Mirror any BGR2RGB or BGR2YUV conversions here
        # TODO - Fix for LSTM
        # Load training video
        train_video = self.load_video(input_video)
        metadata = self.video_metadata(train_video)
        # TODO - Handle blank before / after frames
        num_frames = metadata.get("frames")
        frames = np.arange(0, num_frames)
        train_video = self.preprocess_video(
            train_video, precision, get_frames=frames, **self.input_dims
        )
        total_time = 0
        train_video = np.expand_dims(train_video, axis=0)
        # Manually predict first 2 frames
        start = timer()
        frame_1_2 = model.predict(train_video[:, : self.frames])[:, :2][0]
        end = timer()
        total_time += (end - start) * 1000
        video_size = (num_frames,) + frame_1_2.shape[1:]
        # (frames, height, width, channels)
        predicted_frames = np.zeros(video_size, dtype=precision)
        predicted_frames[:2] = frame_1_2

        # TODO - Use np.delete() if memory issues ?
        # Predict middle frames
        # for i, video_section in enumerate(train_video[0, 2:-2], start=2):
        for i in range(0, num_frames - (self.frames - 1)):
            # print(f"i: {i}")
            # print(f"i+frames: {i+self.frames}")
            # # print(f"Predicting frame {int(i + 1 + self.frames / 2)} of {num_frames}\n")
            # print(f"Predicting frame {int(i + 2 + self.frames / 2)} of {num_frames}\n")
            start = timer()
            # fmt: off
            pred_frame = model.predict(train_video[:, i: i + self.frames])[:, 2]
            # fmt: on
            end = timer()
            predicted_frames[i + 2] = pred_frame
            total_time += (end - start) * 1000

        # Manually predict last 3 frames
        start = timer()
        # fmt: off
        frame_2_1 = model.predict(train_video[:, -self.frames:])[:, -2:][0]
        # fmt: on
        end = timer()
        total_time += (end - start) * 1000
        predicted_frames[-2:] = frame_2_1
        # save_img(
        #     "original.mp4",
        #     self.deprocess_image(np.expand_dims(original_video, axis=0), plot=plot),
        # )
        self.deprocess_video(train_video.squeeze(axis=0), "compressed")

        self.deprocess_video(predicted_frames, "trained")

        original_video = self.load_video(original_video)
        original_video = self.preprocess_video(
            original_video, precision, get_frames=frames, **self.input_dims
        )
        original_video = np.asarray(original_video, dtype=precision)
        self.deprocess_video(original_video, "original")

        return total_time / num_frames

    def deprocess_video(self, video, file_name, file_format="mp4", **kwargs):
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
        # fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(file_name, fourcc, self.fps, (width, height))

        for frame in video:
            frame = self.deprocess_image(frame, **kwargs)
            writer.write(frame)
        # Close video writer
        writer.release()

    @staticmethod
    def get_model_from_string(classname):
        return getattr(sys.modules[__name__].models, classname)

    def load_model_from_path(self, model_path):
        self.out_path = os.sep.join(model_path.split(os.sep)[:-2])
        return load_model(model_path, compile=False)

    @staticmethod
    def loaded_model(model):
        return type(model) == Model

    @staticmethod
    def do_model_saving(model, model_path):
        os.makedirs(model_path)
        os.chdir(model_path)

        # TODO - Use try except (finally) on saving, for memory issues
        # Save the whole model
        model.save(f"{model.name}.h5")  # For Keras (TF 1.0)
        model.save(f"{model.name}.tf")  # TF 2.0

        # Save weights only
        model.save_weights(f"{model.name}_weights.h5")
