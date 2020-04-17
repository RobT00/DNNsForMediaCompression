"""
File containing utility functions for training models and processing data
"""
import os
import sys
import pickle
import models
from datetime import timedelta
from keras.models import load_model, Model
from keras_preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import glob
import cv2
import re
from timeit import default_timer as timer  # Measured in seconds
from tqdm import tqdm


class DataManagement:
    """
    Class used for handling dataflow and processing for training
    """

    def __init__(
        self,
        script_dir: str,
        sequences: bool,
        c_data: str,
        o_data: str,
        out_dir: str,
        input_dims: str,
        c_space: str,
    ):
        """
        Initialise class - generally specified through runtime args in main.py
        :param script_dir: Location of running script used to invoke this class
        :param sequences: Boolean for image (false) or video (true) processing
        :param c_data: Path to compressed (input) data - as training samples
        :param o_data: Path to original (uncompressed) data - as ground truth
        :param out_dir: Directory used for output
        :param input_dims: Dimensions (height, width, channels)
        :param c_space: Colourspace
        """
        self.compare_dict = dict()
        try:
            dims = tuple(map(int, re.findall(r"\d+", input_dims)))
        except TypeError:
            dims = None
        self.input_dims = {"dims": dims} if dims else dict()
        self.precision = "float32"
        self.script_dir = script_dir
        self.sequences = sequences
        self.compressed_data_path = c_data
        self.original_data_path = o_data
        self.out_path = out_dir
        self.frames = 5 if sequences else 1
        self.fps = None
        self.c_space = c_space.upper()

    def preprocess_image(
        self,
        image_path: str,
        mode: dict = None,
        do_conversion: bool = True,
        plot: bool = False,
        **dims,
    ) -> np.ndarray:
        """
        Preprocess images, scale and convert to numpy array for feeding to model.
        :param image_path: Path to image
        :param mode: Mode to process image
        :param do_conversion: Boolean - do conversion to another colourspace
        :param plot: Boolean - to plot image
        :param dims: Image dimensions to be used, tuple - (height, width, channels)
        :return: Processed image
        """
        img = load_img(image_path)
        img = img_to_array(img, dtype=self.precision)
        img = self.check_dims(img, dims.get("dims", img.shape))
        img = self.do_augmentation(mode, img, do_conversion=do_conversion)
        if plot:
            plt.figure()
            plt.imshow(img.astype(float))
            plt.show()

        return img

    def preprocess_video(
        self,
        video_path,
        mode: dict = None,
        do_conversion: bool = True,
        plot: bool = False,
        get_frames: np.ndarray = None,
        **dims,
    ) -> list:
        """
        Preprocess images, scale and convert to numpy array for feeding to model.
        :param video_path: Path to video
        :param mode: Mode to process frames
        :param do_conversion: Boolean - do conversion to other colourspace
        :param plot: Boolean - to show frames
        :param get_frames: Frames to get -- when using generator
        :param dims: Image dimensions to be used, tuple - (height, width, channels)
        :return: Processed frames as list of np.ndarray elements
        """
        if get_frames is not None:
            # Video is already opened and streaming
            cap = video_path
        else:
            cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise UserWarning("Cannot read video, is codec installed?")

        vid = list()
        frames = get_frames if get_frames is not None else range(int(self.frames))
        for i in frames:
            cap.set(1, i)
            ret, frame = cap.read()
            if not ret:
                if get_frames is not None:
                    frame = np.copy(vid[-1])
                    frame.fill(0)
                else:
                    break
            if i < 0:
                frame.fill(0)
            frame = frame.astype(self.precision, copy=False)
            frame = self.check_dims(frame, dims.get("dims", frame.shape))
            frame = self.do_augmentation(
                mode, frame, do_conversion=do_conversion, frame=True
            )
            if plot:
                plt.figure()
                plt.imshow(frame.astype(float))
                # Choice of colourspace may affect how output is represented, matplotlib assumes RGB
                # OpenCV defaults to BGR
                plt.show()
            vid.append(frame)
        cap.release()

        return vid

    def do_augmentation(
        self,
        aug_type: dict,
        img: np.ndarray,
        do_conversion: bool = True,
        frame: bool = False,
    ) -> np.ndarray:
        """
        Function to do some augmentations to an image, diversifying training data
        :param aug_type: Type of augmentation to perform, may be multiple
        :param img: Input image
        :param do_conversion: Do colourspace conversion, if required
        :param frame: If the input image is an OpenCV frame (video)
        :return: Augmented image
        """

        def add_noise(noise_typ: str, image: np.ndarray):
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
                image[tuple(coords)] = 255.0

                # Pepper mode
                num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
                coords = [
                    np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape
                ]
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

        def rotate(degrees: int, image: np.ndarray):
            return ndimage.rotate(image, degrees, reshape=False)

        def brightness(level: float, image: np.ndarray, max_pixel: float = 255.0):
            level *= max_pixel  # Convert percentage to absolute
            return cv2.convertScaleAbs(image, alpha=1.0, beta=level)

        def contrast(level, image):
            return cv2.convertScaleAbs(image, alpha=level, beta=0)

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
                img = np.clip(img, 0.0, 255.0).astype(dtype=self.precision)
        img /= 255.0  # Get floating point values for pixels
        if do_conversion:
            if frame:
                if self.c_space == "YUV":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  # BGR -> YUV
                elif self.c_space == "RGB":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            else:
                if self.c_space == "YUV":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # RGB -> YUV
                elif self.c_space == "BGR":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR
            img = np.clip(img, 0.0, 1.0).astype(dtype=self.precision)

        return img

    @staticmethod
    def get_augemntations() -> dict:
        """
        Using a random number generator, the augmentation(s) to be used are returned as a dictionary
        :return: Dictionary containing augmentations to apply
        """
        noise_types = ["gaussian", "s&p", "poisson", "speckle"]
        # fmt: off
        rotations = [
            1, 5, 10, 15, 20, 30, 40, 45, 55, 60, 75, 90, 120, 150, 160, 180, 270, 290,
        ]  # anti-clockwise
        contrast_values = [
            0.5, 0.75, 0.85, 0.99, 1.0, 1.01, 1.15, 1.25, 1.5,
        ]  # -50% -> +50%
        brightness_values = [
            -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 1, 1.01, 1.2, 1.3, 1.4, 1.5,
        ]  # -50% -> + 50%
        # fmt: on
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
    def load_video(video_path: str) -> cv2.VideoCapture:
        """
        Function for loading a video to the stream, be sure to release the video when done!
        :param video_path: Path to video file
        :return: Video stream
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise UserWarning(f"Cannot read video at {video_path}\nIs codec installed?")

        return video

    def video_metadata(self, video: cv2.VideoCapture) -> dict:
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
    def check_dims(image: np.ndarray, desired_dims: tuple) -> np.ndarray:
        """
        Check if the image dimensions are correct, raise error if not.
        :param image: Tuple of dimensions for image being processed - (height, width, channels)
        :param desired_dims: Tuple of desired image dimensions - (height, width, channels)
        :return: Image with correct dimensions, or raise error
        """

        def check_ok(image_dims: tuple, wanted_dims: tuple) -> bool:
            """
            Check if the image dimensions are the same as the desired dimensions.
            :param image_dims: Tuple of image dimensions - (height, width, channels)
            :param wanted_dims: Tuple of desired image dimensions - (height, width, channels)
            :return: True / False
            """
            return image_dims == wanted_dims

        if not check_ok(image.shape, desired_dims):
            if image.shape[0] != desired_dims[0]:
                if image.shape[1] == desired_dims[0]:
                    image = np.rot90(image)
            if image.shape[2] != desired_dims[2]:
                image = image.reshape([image.shape[0], image.shape[1], desired_dims[2]])
            if not check_ok(image.shape, desired_dims):
                try:
                    # TODO preserve aspect ratio ?
                    image = cv2.resize(
                        image,
                        # (width, height)
                        (desired_dims[1], desired_dims[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                except ValueError:
                    raise ValueError(
                        "Image: {} doesn't match desired dimensions: {}".format(
                            image.shape, desired_dims
                        )
                    )

        return image

    def deprocess_image(
        self,
        img: np.ndarray,
        do_conversion: bool = False,
        frame: bool = False,
        plot: bool = False,
    ) -> np.ndarray:
        """
        Convert the predicted image from the model back to [0..255] for viewing / saving.
        :param img: Predicted image
        :param do_conversion: Boolean - perform colourspace conversion, as required
        :param frame: Specify if input is from a sequence
        :param plot: Boolean - to plot image for viewing
        :return: Restored image
        """
        if img.shape[-1] != 3:
            raise ValueError("Not RGB")
        if len(img.shape) > 3:
            img = img.reshape((img.shape[-3], img.shape[-2], img.shape[-1]))

        if do_conversion:
            if frame:
                if self.c_space == "YUV":
                    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)  # YUV -> BGR
                elif self.c_space == "RGB":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB -> BGR
            else:
                if self.c_space == "YUV":
                    img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)  # YUV -> RGB
                elif self.c_space == "BGR":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            img = np.clip(img, 0.0, None)
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

    def get_input_dims(self) -> tuple:
        """
        Return the desired input dimensions for model, as recorded in the class instance
        :return: Tuple of dimensions (height, width, channels)
        """
        # (height, width, channels)
        if self.sequences:
            # Add number of frames to resolution
            d = self.input_dims.get("dims", (256, 256, 3))
            d = (self.frames,) + d  # Place frames first
        else:
            d = self.input_dims.get("dims", (512, 768, 3))
        return d

    def set_input_dims(self, input_dims: tuple):
        """
        Helper method to set the input dims
        :param input_dims: Tuple of dimensions for input to model
        :return: Updates class with input dims
        """
        self.input_dims.update({"dims": input_dims})

    def generator_function(self, validate: bool = True, split: float = 0.2) -> tuple:
        """
        Perform train/validation split, return appropriate function for training sample generation
        :param validate: Boolean, create a validation set
        :param split: Fraction of data to be used for validation
        :return: Tuple of training data, validation data, function for batch generation
        """
        files = [
            os.path.join(self.compressed_data_path, f)
            for f in os.listdir(self.compressed_data_path)
        ]
        if validate:
            np.random.shuffle(files)
            split_size = int(len(files) * split)
            train_files = files[:-split_size]
            validate_files = files[-split_size:]
        else:
            train_files = files
            validate_files = None
        if self.sequences:
            function = self.video_generator
        else:
            function = self.image_generator
        return train_files, validate_files, function

    def image_generator(
        self, files: list, batch_size: int = 2, file_type: str = "png"
    ) -> tuple:
        """
        Generate training samples for image based models
        :param files: Files to use for mini-batch generating
        :param batch_size: Number of samples in each mini-batch
        :param file_type: Extension of files to load
        :return: model inputs, labels
        """
        self.input_dims.update({"dims": self.get_input_dims()})
        qualities = ["_75.jpg", "_85.jpg", "_90.jpg"]
        # fmt: off
        files = [k for k in files if not any(k[-len(j):] == j for j in qualities)]
        # fmt: on
        while True:
            # Select files for the batch
            batch_paths = np.random.choice(a=files, size=batch_size)
            batch_input = list()
            batch_output = list()

            # Read in each input, perform pre-processing and get labels (original images)
            for input_img in batch_paths:
                augment = self.get_augemntations()
                input = self.preprocess_image(
                    input_img, mode=augment, **self.input_dims
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
                    file_path, mode=augment, **self.input_dims
                )

                batch_input.append(input)
                batch_output.append(output)

            # Return a tuple of (input, output) to feed network
            batch_x = np.asarray(batch_input, dtype=self.precision)
            batch_y = np.asarray(batch_output, dtype=self.precision)

            # return batch_x, batch_y
            yield batch_x, batch_y

    def video_generator(self, files: list, batch_size: int = 2, file_type: str = "y4m"):
        """
        Generate training samples for image based models
        :param files: Files to use for mini-batch generating
        :param batch_size: Number of samples in each mini-batch
        :param file_type: Extension of files to load
        :return: model inputs, labels
        """
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
                # Randomly gather self.frames consecutive frames
                metadata = self.video_metadata(cap)
                # TODO - Handle blank before / after frames
                start_frame = np.random.choice(a=metadata.get("frames") - self.frames)
                frames = np.arange(start_frame, start_frame + self.frames)
                input = self.preprocess_video(
                    cap, get_frames=frames, mode=augment, **self.input_dims
                )
                cap.release()

                # Repeat for output
                re_exp = "(_\d+\.mp4$)"
                file_name = self.get_base_filename(input_video, re_exp)
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
                    cap, get_frames=mid_frame, mode=augment, **self.input_dims
                )
                cap.release()

                batch_input.append(input)
                batch_output.append(output)

            # Return a tuple of (input, output) to feed network
            batch_x = np.asarray(batch_input, dtype=self.precision)
            batch_y = np.asarray(batch_output, dtype=self.precision)

            # return batch_x, batch_y
            yield batch_x, batch_y

    @staticmethod
    def get_base_filename(full_name: str, file_regex: str) -> str:
        """
        Helper function to retrieve a base file name for matching training and label data
        :param full_name: Path to training file
        :param file_regex: Regex to remove from training file to get match
        :return: base name from input path
        """
        re_exp = re.compile(file_regex)
        base_file = os.path.basename(full_name)
        base_filename = re.sub(re_exp, "", base_file)

        return base_filename

    @staticmethod
    def unique_file(dest_path: str) -> str:
        """
        Iterative increase the number on a file name to generate a unique file name
        :param dest_path: Original file name, which may already exist
        :return: Unique file name with appended index, for uniqueness
        """
        index = ""
        # Check if the folder name already exists
        while os.path.exists(index + dest_path):
            if index:
                index = "{}_".format(str(int(index[0:-1]) + 1))
            else:
                index = "1_"

        return index + dest_path

    def output_results(self, model: models, **kwargs) -> timedelta:
        """
        Record metrics from model training, produce samples of model performance
        :param model: instance of trained model
        :param kwargs:
        :return: average time per model output
        """
        if self.sequences:
            return self.output_results_videos(model, **kwargs)
        else:
            return self.output_results_images(model, **kwargs)

    def output_helper(self, model: models, training_data) -> str:
        """
        Utility function for creating required paths for data after training
        Outputs graphs and saves metadata
        :param model: trained model instance
        :param training_data: history from model training
        :return: path to place sample model output
        """
        f_name = ""
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        os.chdir(self.out_path)
        if training_data:
            # Create folder name based on params
            f_name += "optimiser={}_epochs={}_batch_size={}_lr={}".format(
                training_data.model.optimizer.iterations.name.split("/")[0],
                # training_data.params["epochs"],
                len(training_data.epoch),
                training_data.params["batch_size"],
                training_data.params["lr"],
            )

            if self.sequences:
                ms_ssim = "tf_ms_ssim_vid"
                psnr = "tf_psnr_vid"
            else:
                ms_ssim = "tf_ms_ssim"
                psnr = "tf_psnr"

            # Create plots to save training records
            fig_1 = plt.figure()
            plt.plot(
                np.asarray(training_data.history[f"{ms_ssim}"]) * -1.0,
                label=f"MS-SSIM Training Loss",
                color="blue",
            )
            plt.plot(
                np.asarray(training_data.history[f"val_{ms_ssim}"]) * -1.0,
                label=f"MS-SSIM Validation Loss",
                color="orange",
            )
            plt.xlabel("Epochs")
            plt.ylabel("MS-SSIM")
            plt.legend()
            plt.title(f_name)
            # plt.show()

            fig_2 = plt.figure()
            plt.plot(
                np.asarray(training_data.history[f"{psnr}"]) * -1.0,
                label="PSNR Training Loss",
                color="blue",
            )
            plt.plot(
                np.asarray(training_data.history[f"val_{psnr}"]) * -1.0,
                label="PSNR Validation Loss",
                color="orange",
            )
            plt.xlabel("Epochs")
            plt.ylabel("PSNR (dB)")
            plt.legend()
            plt.title(f_name)

            fig_3 = plt.figure()
            plt.plot(
                np.asarray(training_data.history["mean_squared_error"]),
                label="MSE Training Loss",
                color="blue",
            )
            plt.plot(
                np.asarray(training_data.history["val_mean_squared_error"]),
                label="MSE Validation Loss",
                color="orange",
            )
            plt.xlabel("Epochs")
            plt.ylabel("Mean Squared Error")
            plt.legend()
            plt.title(f_name)

            fig_4 = plt.figure()
            plt.plot(
                np.asarray(training_data.history["lr"]),
                label="Learning Rate",
                color="blue",
            )
            plt.xlabel("Epochs")
            plt.ylabel("Learning Rate")
            plt.legend()
            plt.title(f_name)

            out_path = os.path.join(self.out_path, self.unique_file(f_name))

            # Save generated plots
            p_dir = os.path.join(out_path, "Plots")
            os.makedirs(p_dir)
            os.chdir(p_dir)

            fig_1.savefig("MS-SSIM.png")
            fig_2.savefig("PSNR.png")
            fig_3.savefig("MSE.png")
            fig_4.savefig("lr.png")

            # Save model
            m_dir = os.path.join(out_path, "Model")

            self.do_saving(model, training_data, m_dir)

            t_dir = os.path.join(out_path, "Training")
        else:
            t_dir = os.path.join(self.out_path, self.unique_file(f_name))

        return t_dir

    def output_results_images(
        self,
        model: models,
        training_data=None,
        loaded_model: bool = False,
        continue_training: bool = False,
        **kwargs,
    ):
        """
        Produce statistics and samples from training image based model
        :param model: trained model instance
        :param training_data: history of model training
        :param loaded_model: Boolean - specify if model is loaded or trained (new)
        :param continue_training: Boolean - specify if loaded model has been re-trained
        :param kwargs:
        :return: average time for model to produce output
        """
        return_dir = os.getcwd()
        if loaded_model:
            if continue_training:
                dir_name = "trained_model"
            else:
                dir_name = "loaded_model"
            os.chdir(self.out_path)
            self.out_path = os.path.join(self.out_path, self.unique_file(dir_name))
            os.chdir(return_dir)
        else:
            training_dims = f"{model.input_shape[2]}x{model.input_shape[1]}"
            self.out_path = os.path.join(self.out_path, model.name, training_dims)
        t_dir = self.output_helper(model, training_data)

        # Save sample training and validation images
        output_time = 0.0
        i = 0  # So that i will always be defined
        re_exp = r"(_\d+\.jpg$)"
        input_images = os.listdir(self.compressed_data_path)
        num_images = 0
        qualities = sorted(
            set(
                [
                    # fmt: off
                    j[1: len(j) - len(".jpg")]
                    # fmt: on
                    for j in list(set([re.findall(re_exp, i)[0] for i in input_images]))
                ]
            ),
            reverse=True,
        )
        if training_data:
            qualities = qualities[:3]  # get top 3
        for quality in tqdm(qualities, position=0, leave=True):
            match_string = f"_{quality}.jpg"
            for i, image_file in enumerate(
                glob.glob(self.compressed_data_path + f"/*{match_string}"), start=1
            ):
                base_name = self.get_base_filename(image_file, re_exp)
                original_image = os.path.join(
                    self.original_data_path, f"{base_name}.png"
                )
                output_time += self.output_helper_images(
                    t_dir,
                    image_file,
                    original_image,
                    model,
                    output_append=[f"{quality}", base_name],
                )
            num_images += i

        try:
            avg_time = timedelta(milliseconds=output_time / num_images)
        except ZeroDivisionError:
            avg_time = 0.0

        os.chdir(t_dir)
        with open("avg_time.txt", "a") as out_file:
            out_file.write(f"compressed_path: {self.compressed_data_path}\n")
            out_file.write(f"out_path: {self.out_path}\n")
            out_file.write(f"Average time to predict: {avg_time}")

        os.chdir(return_dir)

        return avg_time

    def output_helper_images(
        self,
        output_directory: str,
        input_image: str,
        original_image: str,
        model: models,
        output_append=None,
        plot: bool = False,
    ) -> float:
        """
        Utility function for creating model outputs, used when training or lading model for metrics
        :param output_directory: path to put output
        :param input_image: path of model input
        :param original_image: path of corresponding label
        :param model: trained model
        :param output_append: items to append to output path
        :param plot: Boolean - show outputs
        :return: time (in seconds) to produce model output
        """
        if type(output_append) is list:
            for appendage in output_append:
                output_directory = os.path.join(output_directory, appendage)
        elif output_append:
            output_directory = os.path.join(output_directory, output_append)

        os.makedirs(output_directory)
        os.chdir(output_directory)

        # Load training image
        ndims = len(self.input_dims["dims"])
        dims = {"dims": self.input_dims["dims"][-3:]}
        if ndims == 4:
            seq_len = self.input_dims["dims"][0]
            input_image = self.preprocess_image(input_image, **dims)
            train_image = np.expand_dims(
                np.repeat(np.expand_dims(input_image, axis=0), seq_len, axis=0), axis=0
            )
        else:
            input_image = self.preprocess_image(input_image, **dims)
            train_image = np.expand_dims(input_image, axis=0)

        start = timer()
        pred_image = model.predict(train_image)
        end = timer()

        original_image = self.preprocess_image(
            original_image, do_conversion=False, **dims
        )
        save_img("original.png", self.deprocess_image(original_image, plot=plot))

        save_img(
            "compressed.png",
            self.deprocess_image(input_image, do_conversion=True, plot=plot),
        )

        if ndims == 4:
            try:
                pred_image = pred_image[:, int(seq_len / 2)]
            except IndexError:
                shape = pred_image.shape
                pred_image = np.reshape(pred_image, shape[-3:])
        save_img(
            "trained.png",
            self.deprocess_image(pred_image, do_conversion=True, plot=plot),
        )

        return (end - start) * 1000

    def output_results_videos(
        self,
        model: models,
        training_data: bool = None,
        loaded_model: bool = False,
        continue_training: bool = False,
        **kwargs,
    ) -> float:
        """
        Produce statistics and samples from training image based model
        :param model: trained model instance
        :param training_data: history of model training
        :param loaded_model: Boolean - specify if model is loaded or trained (new)
        :param continue_training: Boolean - specify if loaded model has been re-trained
        :param kwargs:
        :return: average time for model to produce output
        """
        return_dir = os.getcwd()
        encoder = self.compressed_data_path.split(os.sep)[-1]
        low_qual = ""
        no_deblock = ""
        if "LowQual" in self.compressed_data_path.split(os.sep):
            low_qual = "LowQual"
        if "No_deblocking" in self.compressed_data_path.split(os.sep):
            no_deblock = "No_deblocking"
        if loaded_model:
            if continue_training:
                dir_name = f"trained_model_{encoder}"
            else:
                dir_name = f"loaded_model_{encoder}"
            if low_qual:
                dir_name += f"_{low_qual}"
            if no_deblock:
                dir_name += f"_{no_deblock}"
            os.chdir(self.out_path)
            self.out_path = os.path.join(self.out_path, self.unique_file(dir_name))
            os.chdir(return_dir)
        else:
            training_dims = f"{model.input_shape[3]}x{model.input_shape[2]}"
            self.out_path = os.path.join(self.out_path, model.name, encoder)
            if low_qual:
                self.out_path = os.path.join(self.out_path, low_qual)
            if no_deblock:
                self.out_path = os.path.join(self.out_path, no_deblock)
            self.out_path = os.path.join(self.out_path, training_dims)

        t_dir = self.output_helper(model, training_data)

        # Save sample training and validation images
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
        if training_data:
            qualities = qualities[:3]  # get top 3
        for quality in tqdm(qualities, position=0, leave=True):
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
                )
            num_videos += i

        try:
            avg_time = timedelta(milliseconds=output_time / num_videos)
        except ZeroDivisionError:
            avg_time = 0.0

        os.chdir(t_dir)
        with open("avg_time.txt", "a") as out_file:
            out_file.write(f"compressed_path: {self.compressed_data_path}\n")
            out_file.write(f"out_path: {self.out_path}\n")
            out_file.write(f"Average time to predict: {avg_time}")

        os.chdir(return_dir)

        return avg_time

    def output_helper_video(
        self,
        output_directory: str,
        input_video: str,
        original_video: str,
        model: models,
        output_append=None,
        plot=False,
    ) -> float:
        """
        Utility function for creating model outputs, used when training or lading model for metrics
        :param output_directory: path to put output
        :param input_video: path of model input
        :param original_video: path of corresponding label
        :param model: trained model
        :param output_append: items to append to output path
        :param plot: Boolean - show outputs
        :return: average time (in seconds) to produce each output frame
        """
        if type(output_append) is list:
            for appendage in output_append:
                output_directory = os.path.join(output_directory, appendage)
        elif output_append:
            output_directory = os.path.join(output_directory, output_append)

        os.makedirs(output_directory)
        os.chdir(output_directory)
        # Load training video
        train_video = self.load_video(input_video)
        metadata = self.video_metadata(train_video)
        num_frames = metadata.get("frames")
        mid_frame = int(self.frames / 2)
        frames = np.arange(-mid_frame, num_frames + mid_frame)
        train_video = self.preprocess_video(
            train_video, get_frames=frames, **self.input_dims
        )
        total_time = 0.0
        train_video = np.expand_dims(train_video, axis=0)
        video_size = (num_frames,) + self.input_dims["dims"]
        # (frames, height, width, channels)
        predicted_frames = np.zeros(video_size, dtype=self.precision)

        for i in range(num_frames):
            start = timer()
            pred_frame = model.predict(train_video[:, i : i + self.frames])
            end = timer()
            frames_predicted = pred_frame.shape[1]
            if frames_predicted > 1:
                # Not LSTM, multiple output
                pred_frame = pred_frame[:, int(frames_predicted / 2)]
            predicted_frames[i] = pred_frame
            total_time += (end - start) * 1000

        # Use np.delete() if memory issues
        self.deprocess_video(train_video.squeeze(axis=0), "compressed")

        self.deprocess_video(predicted_frames, "trained")

        original_video = self.load_video(original_video)
        original_video = self.preprocess_video(
            original_video, get_frames=frames, do_conversion=False, **self.input_dims
        )
        original_video = np.asarray(original_video, dtype=self.precision)
        self.deprocess_video(original_video, "original", do_conversion=False)

        return total_time / num_frames

    def deprocess_video(
        self,
        video: np.ndarray,
        file_name: str,
        file_format: str = "mkv",
        do_conversion: bool = True,
        **kwargs,
    ):
        """
        Convert the video from a numpy array back to [0..255] for viewing / saving.
        :param video: Frames as numpy array
        :param file_name: Name to save video under
        :param file_format: Format / container to save video as
        :param do_conversion: Boolean - perform colourspace conversion, as required
        :return:
        """
        file_name = file_name + "." + file_format
        width = video.shape[-2]
        height = video.shape[-3]

        # initialise video writer
        # https://stackoverflow.com/questions/51914683/how-to-make-video-from-an-updating-numpy-array-in-python
        # FOURCC Codecs
        # http://www.fourcc.org/codecs.php
        # fourcc = cv2.VideoWriter_fourcc(*"HDYC")
        fourcc = cv2.VideoWriter_fourcc(*"HFYU")  # Use with mkv
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(file_name, fourcc, self.fps, (width, height))

        for frame in video:
            frame = self.deprocess_image(
                frame, do_conversion=do_conversion, frame=True, **kwargs
            )
            writer.write(frame)
        # Close video writer
        writer.release()

    @staticmethod
    def get_model_from_string(classname: str):
        """
        Return constructor for a model from supplied model name
        :param classname: Name of model for instantiation
        :return: Constructor for specified model
        """
        return getattr(sys.modules[__name__].models, classname)

    def load_pickled(self, pickle_file: str = "history") -> dict:
        """
        Load history of a model, saved in pickle format
        :param pickle_file: Name of file to load
        :return: unpickled data
        """
        pickle_path = os.path.join(self.out_path, "Model", f"{pickle_file}.pickle")
        try:
            with open(pickle_path, "rb") as p_file:
                p_data = pickle.load(p_file)
        except FileNotFoundError:
            p_data = dict()
        return p_data

    def load_model_from_path(self, model_path: str):
        """
        Load a trained model from  a file
        :param model_path: Path to saved model file
        :return: Instance of saved model
        """
        self.out_path = os.sep.join(model_path.split(os.sep)[:-2])
        return load_model(model_path, compile=False)

    @staticmethod
    def loaded_model(model: models) -> bool:
        """
        Check if input is an instance of a model
        :param model: Supposed instance of models class to check
        :return: True if model, else false
        """
        return type(model) == Model

    def do_saving(self, model: models, history, model_path: str):
        """
        Save model and metadata post training
        :param model: Trained model instance
        :param history: History of model training
        :param model_path: Path to save model and metadata at
        :return:
        """
        os.makedirs(model_path)
        os.chdir(model_path)

        # Save the whole model
        model.save(f"{model.name}.h5")  # For Keras (TF 1.0)
        model.save(f"{model.name}.tf")  # TF 2.0

        # Save weights only
        model.save_weights(f"{model.name}_weights.h5")

        # Save the history
        with open("history.pickle", "wb") as hist:
            pickle.dump(history.history, hist, protocol=pickle.HIGHEST_PROTOCOL)
        # Save the training params
        with open("params.pickle", "wb") as params:
            history.params["colourspace"] = self.c_space
            pickle.dump(history.params, params, protocol=pickle.HIGHEST_PROTOCOL)
