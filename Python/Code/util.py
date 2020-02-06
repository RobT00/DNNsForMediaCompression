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
from timeit import default_timer as timer  # Measured in seconds


class DataManagement:
    def __init__(self, script_dir, c_images, o_images, o_dir, precision):
        self.compare_dict = dict()
        self.input_dims = dict()
        self.script_dir = script_dir
        self.compressed_images_path = c_images
        self.original_images_path = o_images
        self.out_path = o_dir
        self.train_datagen = ImageDataGenerator(
            rescale=None, dtype=precision, brightness_range=(0.1, 0.9)
        )
        self.test_datagen = ImageDataGenerator(rescale=None, dtype=precision)

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
        if img.shape[3] != 3:
            raise ValueError("Not RGB")
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

    def get_training_images(
        self, precision="float32", img_format="jpg", plot=False, generator=False
    ):
        compressed_images = dict()
        if generator:
            for filename in glob.glob(self.compressed_images_path + f"/*.{img_format}"):
                img = self.preprocess_image(
                    filename, precision, mode="div", plot=plot, **self.input_dims
                )
                if not self.input_dims:
                    self.input_dims.update({"dims": img.shape})
                compression_level = int(
                    os.path.basename(filename).split(img_format)[0].split("_")[1][:-1]
                )
                compressed_images.setdefault(compression_level, list()).append(img)
        else:
            for compression_level in os.listdir(self.compressed_images_path):
                for filename in glob.glob(
                    # fmt: off
                    os.path.join(self.compressed_images_path,
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

    def get_label_images(
        self, num_compressed_images, precision="float32", img_format="png", plot=False
    ):
        original_images = list()
        for filename in glob.glob(self.original_images_path + f"/*.{img_format}"):
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
        return self.input_dims.get("dims", (512, 768, 3))

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

    def output_results(
        self,
        model,
        input_images,
        labels,
        training_data=None,
        precision="float32",
        loss_fn="MS-SSIM",
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
            plt.plot(training_data.history["loss"], label=f"{loss_fn} Training Loss")
            plt.plot(
                training_data.history["val_loss"], label=f"{loss_fn} Validation Loss"
            )
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.legend()
            plt.title(f_name)
            # plt.show()

            fig_2 = plt.figure()
            plt.plot(training_data.history["tf_psnr"], label="PSNR Training Loss")
            plt.plot(training_data.history["val_tf_psnr"], label="PSNR Validation Loss")
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

    @staticmethod
    def get_model_from_string(classname):
        return getattr(sys.modules[__name__].models, classname)

    @staticmethod
    def load_model_from_path(model_path):
        return load_model(model_path, compile=False)

    @staticmethod
    def loaded_model(model):
        return type(model) == Model
