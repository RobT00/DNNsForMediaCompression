"""
File containing utility functions
"""
import os
import sys
from keras.models import load_model, Model
from keras_preprocessing.image import load_img, img_to_array, save_img
from keras_applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import glob
from Python import models


class DataManagement:
    def __init__(self, script_dir, c_images, o_images, o_dir):
        self.compare_dict = dict()
        self.input_dims = dict()
        self.script_dir = script_dir
        self.compressed_images_path = c_images
        self.original_images_path = o_images
        self.out_path = o_dir

    def preprocess_image(
        self,
        image_path,
        precision,
        data_format="channels_first",
        mode="tf",
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
        # img = check_dims(
        #     img,
        #     height=dims.get("height", img.shape[0]),
        #     width=dims.get("width", img.shape[1]),
        #     channels=dims.get("channels", img.shape[2]),
        # )
        img = self.check_dims(img, dims.get("dims", img.shape))
        if mode == "div":
            img /= 255.0
        else:
            img = preprocess_input(img, data_format=data_format, mode=mode)
        if plot:
            plt.figure()
            plt.imshow(img)
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
        img *= 255.0
        img = np.clip(img, 0, 255).astype("uint8")

        if plot:
            plt.figure()
            plt.imshow(img)
            plt.show()

        return img

    def get_training_images(self, precision="float32", img_format="jpg"):
        compressed_images = dict()
        for compression_level in os.listdir(self.compressed_images_path):
            for filename in glob.glob(
                # fmt: off
                os.path.join(self.compressed_images_path,
                             compression_level) + f"/*.{img_format}"
                # fmt: on
            ):
                img = self.preprocess_image(
                    filename, precision, mode="div", plot=False, **self.input_dims
                )
                if not self.input_dims:
                    # dims.update(
                    #     {"height": img.shape[0], "width": img.shape[1], "channels": img.shape[2]}
                    # )
                    self.input_dims.update({"dims": img.shape})
                # if any(c in filename for c in TO_COMPARE):
                #     compare_dict.setdefault("compressed", dict()).setdefault(compression_level, list()).append(img)
                compressed_images.setdefault(compression_level, list()).append(img)
                # im = Image.open(filename)
                # im.load()
                # plt.figure()
                # plt.imshow(im)
                # plt.show()
                # compressed_images.append(np.asarray(im, dtype="int32"))
                #
                # img = load_img(filename)
                # img = img_to_array(img)
                # # img = np.expand_dims(img, axis=0)
                # # o_img = np.copy(img)
                # # img = inception_v3.preprocess_input(img)
                # img = preprocess_input(img, data_format='channels_first', mode='tf')  # or data_format='channels_last'
                # plt.figure()
                # plt.imshow(img)
                # plt.show()
                # test_list.append(img)

        compressed_images_array = list()
        for i in compressed_images.values():
            compressed_images_array.extend(i)

        compressed_images_array = np.asarray(compressed_images_array, dtype=precision)

        return compressed_images, compressed_images_array

    def get_label_images(
        self, num_compressed_images, precision="float32", img_format="png"
    ):
        original_images = list()
        for filename in glob.glob(self.original_images_path + f"/*.{img_format}"):
            img = self.preprocess_image(
                filename, precision, mode="div", plot=False, **self.input_dims
            )
            if not self.input_dims:
                # dims.update(
                #     {"height": img.shape[0], "width": img.shape[1], "channels": img.shape[2]}
                # )
                self.input_dims.update({"dims": img.shape})
            original_images.append(img)
            # im = Image.open(filename)
            # im.load()
            # original_images.append(np.asarray(im, dtype="int32"))

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

    def output_results(self, model, input_images, labels, training_data=None):
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
                training_data.params["epochs"],
                training_data.params["batch_size"],
            )

            # Create plots to save training records
            fig_1 = plt.figure()
            plt.plot(training_data.history["loss"], label="PSNR Training Loss")
            plt.plot(training_data.history["val_loss"], label="PSNR Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.legend()
            plt.title(f_name)
            # plt.show()

            fig_2 = plt.figure()
            plt.plot(training_data.history["mse"], label="MSE Training Loss")
            plt.plot(training_data.history["val_mse"], label="MSE Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.legend()
            plt.title(f_name)
            # plt.show()

            # plt.figure()
            # plt.plot(history.history['loss'], label="MSE Training Loss")
            # plt.plot(history.history['val_loss'], label="MSE Validation Loss")
            # plt.plot(history.history['rmse'], label="RMSE Training Loss")
            # plt.plot(history.history['val_rmse'], label="RMSE Validation Loss")
            # plt.xlabel("Epochs")
            # plt.ylabel("Score")
            # plt.legend()
            # plt.show()
            #
            # plt.figure()
            # plt.plot(history.history['tf_psnr'], label="PSNR Training Loss")
            # plt.plot(history.history['val_tf_psnr'], label="PSNR Validation Loss")
            # plt.xlabel("Epochs")
            # plt.ylabel("Score")
            # plt.legend()
            # plt.show()

            f_name += " metrics={} model={}".format(
                ",".join(training_data.params["metrics"]), model.name
            )

            # out_path = self.unique_file(os.path.join(self.out_path, f_name))
            out_path = os.path.join(self.out_path, self.unique_file(f_name))

            # Save generated plots
            p_dir = os.path.join(out_path, "Plots")
            os.makedirs(p_dir)
            os.chdir(p_dir)

            fig_1.savefig("PSNR.png")
            fig_2.savefig("RMSE.png")

            # Save model
            m_dir = os.path.join(out_path, "Model")
            os.makedirs(m_dir)
            os.chdir(m_dir)

            model.save("{}.h5".format(model.name))
        else:
            f_name += "loaded_model={}".format(model.name)
            # out_path = self.unique_file(os.path.join(self.out_path, f_name))
            out_path = os.path.join(self.out_path, self.unique_file(f_name))

        # Save sample training and validation images
        t_dir = os.path.join(out_path, "Training")
        if type(input_images) is dict:
            for compression_level, images in input_images.items():
                for i, (t_im, o_im) in enumerate(zip(images, labels)):
                    self.output_helper_images(
                        t_dir,
                        t_im,
                        o_im,
                        model,
                        output_append=[str(compression_level), str(i)],
                    )
        else:
            for i, (t_im, o_im) in enumerate(zip(input_images, labels)):
                self.output_helper_images(
                    t_dir, t_im, o_im, model, output_append=str(i)
                )

        # os.chdir(self.script_dir)
        os.chdir(return_dir)

    def output_helper_images(
        self, output_directory, input_image, original_image, model, output_append=None
    ):
        if type(output_append) is list:
            for appendage in output_append:
                output_directory = os.path.join(output_directory, appendage)
        elif output_append:
            output_directory = os.path.join(output_directory, output_append)

        os.makedirs(output_directory)
        os.chdir(output_directory)

        train_im = np.expand_dims(input_image, axis=0)
        train_pred = model.predict(train_im)
        save_img(
            "original.png",
            self.deprocess_image(np.expand_dims(original_image, axis=0), plot=False),
        )
        save_img("compressed.png", self.deprocess_image(train_im, plot=False))
        save_img("trained.png", self.deprocess_image(train_pred, plot=False))

    @staticmethod
    def get_model_from_string(classname):
        return getattr(sys.modules[__name__].models, classname)

    @staticmethod
    def load_model_from_path(model_path):
        return load_model(model_path, compile=False)

    @staticmethod
    def loaded_model(model):
        return type(model) == Model
