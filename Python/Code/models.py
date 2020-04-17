"""
File for all models, used by main.py
"""
import os
import ipykernel
import numpy as np
import datetime
from keras import backend as K
from timeit import default_timer as timer
from keras import Model, Input
from keras.layers import (
    Conv2D,
    Conv3D,
    Conv2DTranspose,
    Conv3DTranspose,
    Dropout,
    MaxPooling2D,
    MaxPooling3D,
    AveragePooling2D,
    AveragePooling3D,
    UpSampling2D,
    UpSampling3D,
    ZeroPadding2D,
    ZeroPadding3D,
    GaussianDropout,
    Cropping2D,
    concatenate,
    Reshape,
    ConvLSTM2D,
    Lambda,
)
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.callbacks import TimeStopping
from sklearn.model_selection import train_test_split


class ModelClass:
    """
    Base class used for instantiating models
    Forms basis of functions models use
    """

    def __init__(
        self, dims: tuple, precision: str = "float32", c_space: str = "YUV", **kwargs
    ):
        """
        Create base attributes for model instance
        :param dims: Expected input dimensions for model
        :param precision: Floating point precision of model
        :param c_space: Colourspace model operates in
        :param kwargs:
        """
        self.set_precision(precision)
        self.input = self.input_layer(dims)
        self.c_space = c_space

    @staticmethod
    def input_layer(dims: tuple) -> Input:
        """
        Create input layer for model
        :param dims: Expected input dimensions for model
        :return: Input layer for model
        """
        return Input(shape=dims, name="input")

    @staticmethod
    def set_precision(precision: str = "float32"):
        """
        Set floating point precision in Keras backend
        :param precision: Floating point precision to use
        :return:
        """
        K.set_floatx(precision)

    def train(
        self, model, run_epochs: int = 1, batch_size: int = 2, util_class=None, **kwargs
    ):
        """
        Function to set up and perform model training
        :param model: Model to be trained
        :param run_epochs: Epochs to train for
        :param batch_size: Mini-batch size to use
        :param util_class: Instance of DataManagement class model is using
        :param kwargs:
        :return: History of trained model
        """
        verbosity = 1
        max_time_seconds = int(60 * 60 * 16.5)
        monitor_metric = "val_mean_squared_error"
        cb = list()
        cb_patience = min(int(run_epochs * 0.15), 20)
        cb.append(
            ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=0.1,
                patience=cb_patience,
                verbose=0,
                mode="min",
                min_delta=1e-4,
                cooldown=0,
                min_lr=0,
                **kwargs,
            )
        )

        if run_epochs > 100:
            cb_patience = min(int(run_epochs * 0.2), 50)
            cb.append(
                EarlyStopping(
                    verbose=True,
                    patience=cb_patience,
                    monitor=monitor_metric,
                    mode="min",
                    min_delta=1e-5,
                )
            )
        cb.append(TimeStopping(seconds=max_time_seconds, verbose=verbosity))
        print("Training")
        start = timer()

        test_files, val_files, gen_function = util_class.generator_function()
        if util_class.sequences:
            epoch_steps = (len(test_files) * util_class.frames * 30) // batch_size
            val_steps = (len(val_files) * util_class.frames * 30) // batch_size
        else:
            # epoch_steps = 144 // batch_size
            epoch_steps = 408 // batch_size
            val_steps = epoch_steps // batch_size
        history = model.fit_generator(
            gen_function(test_files, batch_size=batch_size, **kwargs),
            steps_per_epoch=epoch_steps,
            epochs=run_epochs,
            verbose=verbosity,
            callbacks=cb,
            validation_data=gen_function(val_files, batch_size=batch_size, **kwargs),
            validation_steps=val_steps,
            validation_freq=1,
            class_weight=None,
            max_queue_size=15,
            workers=1,
            use_multiprocessing=False,
            shuffle=True,
            initial_epoch=0,
        )

        end = timer()
        dur = end - start
        print("Training took: {}".format(str(datetime.timedelta(seconds=dur))))

        return history

    @staticmethod
    def lr_schedule(epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print("Learning rate: ", lr)
        return lr

    @staticmethod
    def crop(dim, start, end, **kwargs):
        """
        Crops (or slices) a Tensor on a given dimension from start to end
        example : to crop tensor x[:, :, 5:10]
        :param dim: dimension to augment
        :param start:
        :param end:
        :param kwargs: kwargs for further functionality
        :return: Cropped tensor
        """

        def func(x):
            dimension = dim
            if dimension == -1:
                dimension = len(x.shape) - 1
            if dimension == 0:
                return x[start:end]
            if dimension == 1:
                return x[:, start:end]
            if dimension == 2:
                return x[:, :, start:end]
            if dimension == 3:
                return x[:, :, :, start:end]
            if dimension == 4:
                return x[:, :, :, :, start:end]

        return Lambda(func, **kwargs)


"""
IMAGE MODELS
"""


class Attempt1(ModelClass):
    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(
            dims, precision, **kwargs
        )  # Is equivalent to super(Attempt1, self).__init__(dims)
        self.name = "Attempt1"

    def build(self):
        x = Conv2D(filters=64, kernel_size=3, activation="relu")(self.input)
        x = ZeroPadding2D(padding=(2, 2))(x)
        x = Conv2D(filters=32, kernel_size=3, activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=3, activation="tanh")(x)
        x = Dropout(0.05)(x)
        encode = Conv2D(filters=3, kernel_size=2, strides=(2, 2))(x)

        # model.add(Flatten())
        x = UpSampling2D(size=(2, 2))(encode)
        x = Conv2DTranspose(filters=8, kernel_size=2, strides=(2, 2), padding="valid")(
            x
        )
        x = Conv2DTranspose(
            filters=16,
            kernel_size=3,
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=regularizers.l2(0.01),
        )(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(filters=32, kernel_size=3, strides=(1, 1), padding="same")(
            x
        )
        x = ZeroPadding2D(padding=(2, 2))(x)
        x = GaussianDropout(0.02)(x)
        x = Conv2DTranspose(
            filters=16, kernel_size=3, activation="relu", padding="same"
        )(x)
        x = Conv2DTranspose(filters=8, kernel_size=5, strides=(1, 1), padding="same")(x)
        decode = Conv2D(filters=3, kernel_size=2, strides=(2, 2), padding="valid")(x)
        # model.add(Reshape((512, 768, 3)))

        model = Model(self.input, decode)

        model.name = self.name

        model.c_space = self.c_space

        return model


class Attempt2(ModelClass):
    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(
            dims, precision, **kwargs
        )  # Is equivalent to super(Attempt2, self).__init__(dims)
        self.name = "Attempt2"

    def build(self):
        x = Conv2D(filters=32, kernel_size=3, activation="relu", padding="valid")(
            self.input
        )  # (x-2, y-2, 32) -> (510, 766, 32)
        x = UpSampling2D(size=(2, 2))(x)  # ((x*2), (y*2), 32) -> (1020, 1532, 32)
        x = Conv2D(filters=64, kernel_size=2, activation="relu", padding="valid")(
            x
        )  # ((x-1), (y-1), 64) -> (1019, 1531, 64)
        x = Conv2DTranspose(
            filters=42, kernel_size=2, activation="relu", padding="valid"
        )(
            x
        )  # ((x+1), (y+1), 42) -> (1020, 1532, 42)
        x = ZeroPadding2D(padding=(2, 2))(x)  # ((x+4), (y+4), 42) -> (1024, 1536, 42)
        upscale_encode = Conv2DTranspose(
            filters=69,
            kernel_size=3,
            activation="tanh",
            kernel_regularizer=regularizers.l2(0.01),
            padding="same",
            strides=(2, 2),
            name="upscale_encode",
        )(
            x
        )  # ((x*2), (y*2), 69) -> (2048, 3072, 69) [3072 / 2048 = 1.5]

        x = Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(
            upscale_encode
        )  # (x, y, 32) -> (2048, 3072, 32)
        x = MaxPooling2D(pool_size=(2, 2))(x)  # (x/2, y/2, 32) -> (1024, 1536, 32)
        x = Conv2D(filters=128, kernel_size=3, activation="tanh", padding="valid")(
            x
        )  # (x-2, y-2, 128) -> (1022, 1534, 128)
        x = Dropout(0.05)(x)  # (x, y, 128) -> (1022, 1534, 128)
        x = Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(
            x
        )  # (x, y, 32) -> (1022, 1534, 32)
        # x = MaxPooling2D(pool_size=(3, 3))(x)  # (x/3, y/3, 32) -> (340, 511, 32)
        x = Conv2D(
            filters=12,
            kernel_size=3,
            strides=(1, 1),
            activation="tanh",
            padding="valid",
        )(
            x
        )  # ((x-2), (y-2), z) -> (1020, 1532, 12)
        x = AveragePooling2D(pool_size=(2, 2))(x)  # (x/2, y/2, z) -> (510, 766, 12)
        x = Conv2D(
            filters=24,
            kernel_size=5,
            strides=(2, 2),
            activation="relu",
            padding="valid",
        )(
            x
        )  # ((x-4)/2, (y-4)/2, z) -> (253, 381, 24)
        x = Conv2D(
            filters=32,
            kernel_size=2,
            strides=(2, 2),
            activation="relu",
            padding="valid",
        )(
            x
        )  # ((x-1)/2, (y-1)/2, z) -> (126, 190, 32)
        x = Conv2D(
            filters=42,
            kernel_size=3,
            strides=(1, 1),
            activation="tanh",
            kernel_regularizer=regularizers.l1(0.01),
            padding="valid",
        )(
            x
        )  # ((x-2)/1, (y-2)/1, z) -> (124, 188, 42)
        x = ZeroPadding2D(padding=(1, 0))(x)  # (x+(1*2), y+(0*2), z) -> (126, 188, 42)
        x = GaussianDropout(0.2)(x)
        encode = Conv2D(
            filters=64,
            kernel_size=3,
            strides=(1, 1),
            activation="relu",
            padding="valid",
            name="encode",
        )(
            x
        )  # (x-2, y-2, z) -> (124, 186, 64) [186 / 124 = 1.5]

        # model.add(Flatten())
        x = Conv2DTranspose(filters=32, kernel_size=2, strides=(1, 1), padding="same")(
            encode
        )  # (x, y, z) -> (124, 186, 32)
        x = Conv2DTranspose(
            filters=8,
            kernel_size=3,
            strides=(2, 2),
            output_padding=1,
            dilation_rate=2,
            padding="valid",
        )(
            x
        )  # ((x+2)*2, (y+2)*2, z) -> (252, 376, 8)
        # x = UpSampling2D(size=(2, 2))(x)  # (x*2, y*2, 3) -> (504, 752, 8)
        x = Conv2DTranspose(
            filters=16,
            kernel_size=3,
            strides=(2, 2),
            padding="valid",
            kernel_regularizer=regularizers.l2(0.01),
            output_padding=1,
            dilation_rate=2,
        )(
            x
        )  # ((x+2)*2, (y+2)*2, z) -> (508, 756, 16)
        # # x = UpSampling2D(size=(2, 2))(x)  # (x*2, y*2, z) -> (2988, 4476, 16)
        # x = Conv2DTranspose(filters=32, kernel_size=3, strides=(1, 1), padding="valid")(
        #     x
        # )  # ((x+2)*1, (y+2)*1, z) -> (1346, 2034, 32)
        # # x = ZeroPadding2D(padding=(2, 2))(x)  # (x+(2*2), y+(2*2), z) -> (1342, 2030, 32)
        # x = GaussianDropout(0.02)(x)  # (x, y, z) -> (1342, 2030, 32)
        # x = Conv2DTranspose(
        #     filters=64, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
        # )(
        #     x
        # )  # (x*2, y*2, z) -> (1018, 1530, 64)
        x = ZeroPadding2D(padding=(0, 3))(x)  # (x+(0*2), y+(3*2), z) -> (508, 762, 16)
        x = Conv2DTranspose(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            activation="tanh",
            kernel_regularizer=regularizers.l1(0.01),
            padding="valid",
            output_padding=1,
            dilation_rate=2,
        )(
            x
        )  # ((x+2)*2, (y+2)*2, z) -> (1020, 1528, 128)
        x = ZeroPadding2D(padding=(1, 3))(
            x
        )  # (x+(1*2), y+(9*2), z) -> (1022, 1534, 128)
        upscale_decode = Conv2DTranspose(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="valid",
            name="upscale_decode",
        )(
            x
        )  # ((x+2)*1, (y+2)*1, z) -> (1024, 1536, 32) [1536 / 1024 = 1.5]

        # x = ZeroPadding2D(padding=(2, 3))(upscaler)  # (x+(2*2), y+(3*2), z) -> (8058, 12188, 16)
        # x = Conv2DTranspose(filters=8, kernel_size=5, strides=(1, 1), padding="same")(x)
        # (x*1, y*1, z) -> (8058, 12188, 8)
        # x = Conv2D(filters=12, kernel_size=2, strides=(2, 2), padding="valid")(
        #     upscale_decode
        # )  # ((x-1)/2, (y-1)/2, z) -> (12345, 18489, 12)
        x = Conv2D(
            filters=8, kernel_size=5, strides=(1, 1), activation="relu", padding="valid"
        )(
            upscale_decode
        )  # ((x-4)/1, (y-4)/1, z) -> (1020, 1532, 8)
        x = GaussianDropout(0.1)(x)  # (x, y, z) -> (1020, 1532, 8)
        x = Conv2D(
            filters=6, kernel_size=2, strides=(2, 2), activation="relu", padding="valid"
        )(
            x
        )  # ((x-1)/2, (y-1)/2, 6) -> (510, 766, 6)
        # x = Conv2D(
        #     filters=4, kernel_size=3, strides=(2, 2), padding="valid"
        # )(x)  # ((x-2)/2, (y-2)/2, z) -> (513, 769, 4)
        x = ZeroPadding2D(padding=(2, 2))(x)  # (x+(2*2), y+(2*2), z) -> (514, 770, 6)
        decode = Conv2D(
            filters=3,
            kernel_size=3,
            strides=(1, 1),
            activation="tanh",
            padding="valid",
            name="decode",
        )(
            x
        )  # ((x-2)/1, (y-2)/1, z) -> (512, 768, 3) [768 / 512 = 1.5]
        # model.add(Reshape((512, 768, 3)))

        model = Model(self.input, decode)

        model.name = self.name

        model.c_space = self.c_space

        return model


class Attempt3(ModelClass):
    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(
            dims, precision, **kwargs
        )  # Is equivalent to super(Attempt3, self).__init__(dims)
        self.name = "Attempt3"

    def build(self):
        # x = Conv2D(filters=32, kernel_size=3, activation="relu", padding="valid")(
        #     self.input
        # )  # (x-2, y-2, 32) -> (510, 766, 32)
        # x = UpSampling2D(size=(2, 2))(x)  # ((x*2), (y*2), 32) -> (1020, 1532, 32)
        # x = Conv2D(filters=64, kernel_size=2, activation="relu", padding="valid")(
        #     x
        # )  # ((x-1), (y-1), 64) -> (1019, 1531, 64)
        x = Conv2DTranspose(
            filters=8, kernel_size=(3, 4), activation="relu", padding="valid"
        )(
            self.input
        )  # ((x+2), (y+2), z) -> (514, 770, 8)
        # x = ZeroPadding2D(padding=(0, 1))(x)  # (x+(0*2), y+(1*2), z) -> (514, 772, 8)
        upscale_encode = Conv2DTranspose(
            filters=16,
            kernel_size=2,
            activation="tanh",
            kernel_regularizer=regularizers.l2(0.01),
            padding="valid",
            strides=(2, 2),
            name="upscale_encode",
            # output_padding=(0, 1),
            # dilation_rate=1
        )(
            x
        )  # ((x+2)*2, (y+2)*2, z) -> (1028, 1542, 16) [1536 / 1024 = 1.5]

        x = Conv2D(filters=24, kernel_size=3, activation="relu", padding="same")(
            upscale_encode
        )  # (x, y, z) -> (1028, 1542, 32)
        x = Conv2D(
            filters=32,
            kernel_size=5,
            strides=(2, 2),
            activation="tanh",
            padding="valid",
        )(
            x
        )  # ((x-4)/2, (y-4)/2, z) -> (512, 769, 42)
        x = MaxPooling2D(pool_size=(2, 2))(x)  # (x/2, y/2, z) -> (256, 384, 42)
        x = Dropout(0.05)(x)  # (x, y, z) -> (256, 384, 42)
        x = Conv2D(filters=42, kernel_size=3, activation="relu", padding="valid")(
            x
        )  # (x-2, y-2, z) -> (254, 382, 48)
        # x = MaxPooling2D(pool_size=(3, 3))(x)  # (x/3, y/3, 32) -> (340, 511, 32)
        x = Conv2D(
            filters=48,
            kernel_size=3,
            strides=(2, 2),
            activation="tanh",
            padding="valid",
        )(
            x
        )  # ((x-2)/2, (y-2)/2, z) -> (126, 190, 56)
        x = AveragePooling2D(pool_size=(2, 2))(x)  # (x/2, y/2, z) -> (63, 95, 56)
        x = Conv2D(
            filters=56,
            kernel_size=5,
            strides=(2, 2),
            activation="relu",
            padding="valid",
        )(
            x
        )  # ((x-4)/2, (y-4)/2, z) -> (30, 46, 24)
        # x = Conv2D(
        #     filters=32,
        #     kernel_size=2,
        #     strides=(2, 2),
        #     activation="relu",
        #     padding="valid",
        # )(
        #     x
        # )  # ((x-1)/2, (y-1)/2, z) -> (126, 190, 32)
        # x = Conv2D(
        #     filters=42,
        #     kernel_size=3,
        #     strides=(1, 1),
        #     activation="tanh",
        #     kernel_regularizer=regularizers.l1(0.01),
        #     padding="valid",
        # )(
        #     x
        # )  # ((x-2)/1, (y-2)/1, z) -> (124, 188, 42)
        x = ZeroPadding2D(padding=(2, 2))(x)  # (x+(1*2), y+(0*2), z) -> (34, 50, 42)
        x = GaussianDropout(0.2)(x)
        encode = Conv2D(
            filters=64,
            kernel_size=3,
            strides=(1, 1),
            activation="relu",
            padding="valid",
            name="encode",
        )(
            x
        )  # (x-2, y-2, z) -> (32, 48, 32) [48 / 32 = 1.5]
        # 32 * 2 * 2 * 2 * 2 = 512
        # model.add(Flatten())
        x = Conv2DTranspose(filters=32, kernel_size=2, strides=(2, 2), padding="valid")(
            encode
        )  # ((x+1)*2 - 2, (y+1)*2 - 2, z) -> (64, 96, 32)  (64, 98, 32)
        # x = Conv2DTranspose(
        #     filters=8,
        #     kernel_size=3,
        #     strides=(2, 2),
        #     output_padding=1,
        #     dilation_rate=2,
        #     padding="valid",
        # )(
        #     x
        # )  # ((x+2)*2, (y+2)*2, z) -> (252, 376, 8)
        # x = UpSampling2D(size=(2, 2))(x)  # (x*2, y*2, 3) -> (504, 752, 8)
        x = Conv2DTranspose(
            filters=16,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            kernel_regularizer=regularizers.l2(0.01),
            # output_padding=1,
            # dilation_rate=2,
            name="issue",
        )(
            x
        )  # ((x+2)*2, (y+2)*2, z) -> (128, 192, 16) (128, 196, 16)
        # # x = UpSampling2D(size=(2, 2))(x)  # (x*2, y*2, z) -> (2988, 4476, 16)
        # x = Conv2DTranspose(filters=32, kernel_size=3, strides=(1, 1), padding="valid")(
        #     x
        # )  # ((x+2)*1, (y+2)*1, z) -> (1346, 2034, 32)
        # # x = ZeroPadding2D(padding=(2, 2))(x)  # (x+(2*2), y+(2*2), z) -> (1342, 2030, 32)
        # x = GaussianDropout(0.02)(x)  # (x, y, z) -> (1342, 2030, 32)
        # x = Conv2DTranspose(
        #     filters=64, kernel_size=3, strides=(2, 2), activation="relu", padding="same"
        # )(
        #     x
        # )  # (x*2, y*2, z) -> (1018, 1530, 64)
        # x = ZeroPadding2D(padding=(0, 2))(x)  # (x+(0*2), y+(2*2), z) -> (128, 196, 16)
        x = Conv2DTranspose(
            filters=8,
            kernel_size=3,
            strides=(2, 2),
            activation="tanh",
            kernel_regularizer=regularizers.l1(0.01),
            padding="valid",
            output_padding=1,
        )(
            x
        )  # ((x+2)*2, (y+2)*2, z) -> (258, 394, 8)  (256, 392, 8)
        # x = ZeroPadding2D(padding=(1, 3))(
        #     x
        # )  # (x+(1*2), y+(9*2), z) -> (1022, 1534, 128)
        x = Conv2DTranspose(
            filters=5,
            kernel_size=2,
            strides=(2, 2),
            activation="relu",
            padding="valid",
            name="upscale_decode",
        )(
            x
        )  # ((x+1)*2, (y+1)*2, z) -> (516, 772, 5) [768 / 512 = 1.5]
        decode = Conv2D(
            filters=3,
            kernel_size=5,
            strides=(1, 1),
            activation="relu",
            kernel_regularizer=regularizers.l2(0.02),
            name="decode",
            padding="valid",
        )(
            x
        )  # (x-4, y-4, z) -> (512, 768, 3)

        # # x = ZeroPadding2D(padding=(2, 3))(upscaler)  # (x+(2*2), y+(3*2), z) -> (8058, 12188, 16)
        # # x = Conv2DTranspose(filters=8, kernel_size=5, strides=(1, 1), padding="same")(x)
        # # (x*1, y*1, z) -> (8058, 12188, 8)
        # # x = Conv2D(filters=12, kernel_size=2, strides=(2, 2), padding="valid")(
        # #     upscale_decode
        # # )  # ((x-1)/2, (y-1)/2, z) -> (12345, 18489, 12)
        # x = Conv2D(
        #     filters=8, kernel_size=5, strides=(1, 1), activation="relu", padding="valid"
        # )(
        #     upscale_decode
        # )  # ((x-4)/1, (y-4)/1, z) -> (1020, 1532, 8)
        # x = GaussianDropout(0.1)(x)  # (x, y, z) -> (1020, 1532, 8)
        # x = Conv2D(
        #     filters=6, kernel_size=2, strides=(2, 2), activation="relu", padding="valid"
        # )(
        #     x
        # )  # ((x-1)/2, (y-1)/2, 6) -> (510, 766, 6)
        # # x = Conv2D(
        # #     filters=4, kernel_size=3, strides=(2, 2), padding="valid"
        # # )(x)  # ((x-2)/2, (y-2)/2, z) -> (513, 769, 4)
        # x = ZeroPadding2D(padding=(2, 2))(x)  # (x+(2*2), y+(2*2), z) -> (514, 770, 6)
        # decode = Conv2D(
        #     filters=3,
        #     kernel_size=3,
        #     strides=(1, 1),
        #     activation="tanh",
        #     padding="valid",
        #     name="decode",
        # )(
        #     x
        # )  # ((x-2)/1, (y-2)/1, z) -> (512, 768, 3) [768 / 512 = 1.5]
        # # model.add(Reshape((512, 768, 3)))

        model = Model(self.input, decode)

        model.name = self.name

        model.c_space = self.c_space

        return model


class Attempt4(ModelClass):
    """
    Model class for autoencoder attempt - fully symmetric, using only centred kernel (i.e. 3x3, 5x5);
    Upscale once (2x), then encode to latent space
    Decode to (2x) then convolve to output image
    """

    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(dims, precision, **kwargs)
        self.name = "Attempt4"

    def build(self):
        x = Conv2DTranspose(
            filters=3,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            name="input_upscale_1",
        )(
            self.input
        )  # (1024, 1536, 3)
        x = Conv2D(
            filters=8,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            name="input_encode_1",
        )(
            x
        )  # (512, 768, 8)
        x = Conv2D(
            filters=16,
            kernel_size=5,
            strides=(2, 2),
            padding="valid",
            name="input_encode_2",
        )(
            x
        )  # (254, 382, 16)
        x = MaxPooling2D(pool_size=(2, 2))(x)  # (127, 191, 16)
        x = Conv2D(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            name="input_encode_3",
        )(
            x
        )  # (64, 96, 32)
        # x = ZeroPadding2D(padding=(1, 0))(x)  # (66, 96, 32)
        encode = Conv2D(
            filters=64, kernel_size=3, padding="valid", name="input_encode_4"
        )(
            x
        )  # (62, 94, 64)

        x = Conv2DTranspose(
            filters=48, kernel_size=3, padding="same", name="output_decode_1"
        )(
            encode
        )  # (62, 94, 48)
        x = ZeroPadding2D(padding=(1, 1))(x)
        # x = UpSampling2D(size=(2, 2))(x)  # (128, 188, 48)
        # x = Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="valid", output_padding=1,
        # dilation_rate=2, name="output_decode_2")(x)  # (260, 380, 32)
        x = Conv2DTranspose(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            name="output_decode_2",
        )(
            x
        )  # (124, 188, 32)
        x = UpSampling2D(size=(2, 2))(x)  # (248, 376, 32)
        # x = Conv2DTranspose(filters=16, kernel_size=5, strides=(2, 2), padding="valid", output_padding=1,
        # dilation_rate=2, name="output_decode_3")(x)  # (509, 765, 16)
        x = Conv2DTranspose(
            filters=16,
            kernel_size=5,
            strides=(2, 2),
            padding="valid",
            name="output_decode_3",
        )(
            x
        )  # (504, 760, 16)
        x = Conv2DTranspose(
            filters=8,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            name="output_upscale_1",
        )(
            x
        )  # (1008, 1520, 8)
        x = Cropping2D(cropping=((2, 2), (2, 2)))(x)
        decode = Conv2D(
            filters=3,
            kernel_size=3,
            strides=(2, 2),
            padding="valid",
            name="output_decode_4",
        )(
            x
        )  # (512, 768, 3)
        # x = Cropping2D(cropping=((2, 1), (2, 1)))(x)
        # decode = Conv2D(filters=3, kernel_size=3, padding="valid", name="output_decode_5")(x)

        model = Model(self.input, decode)

        model.name = self.name

        model.c_space = self.c_space

        return model


class KerasAE(ModelClass):
    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(
            dims, precision, **kwargs
        )  # Is equivalent to super(KerasAE, self).__init__(dims)
        self.name = "Keras CNN AE"

    def build(self):
        x = Conv2D(16, (3, 3), activation="relu", padding="same")(self.input)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
        encoded = MaxPooling2D((2, 2), padding="same")(x)

        x = Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D((2, 2))(x)
        decode = Conv2D(3, (3, 3), activation="tanh", padding="same")(x)

        model = Model(self.input, decode)

        model.name = self.name

        model.c_space = self.c_space

        return model


class KerasDenoise(ModelClass):
    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(
            dims, precision, **kwargs
        )  # Is equivalent to super(KerasDenoise, self).__init__(dims)
        self.name = "Keras Denoise AE"

    def build(self):
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(self.input)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        encoded = MaxPooling2D((2, 2), padding="same")(x)

        x = Conv2D(32, (3, 3), activation="relu", padding="same")(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D((2, 2))(x)
        decode = Conv2D(3, (3, 3), activation="relu", padding="same")(x)

        model = Model(self.input, decode)

        model.name = self.name

        model.c_space = self.c_space

        return model


class UNet(ModelClass):
    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(dims, precision, **kwargs)
        self.name = "U-Net"

    def build(self):
        conv1 = Conv2D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(self.input)
        conv1 = Conv2D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool1)
        conv2 = Conv2D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool2)
        conv3 = Conv2D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool3)
        conv4 = Conv2D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(
            1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool4)
        conv5 = Conv2D(
            1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(
            512, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge6)
        conv6 = Conv2D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv6)

        up7 = Conv2D(
            256, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge7)
        conv7 = Conv2D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv7)

        up8 = Conv2D(
            128, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge8)
        conv8 = Conv2D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv8)

        up9 = Conv2D(
            64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge9)
        conv9 = Conv2D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv9)
        decode = Conv2D(
            3, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv9)

        model = Model(self.input, decode)

        model.name = self.name

        model.c_space = self.c_space

        return model


"""
VIDEO MODELS
"""


class Attempt1_3D(ModelClass):
    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(
            dims, precision, **kwargs
        )  # Is equivalent to super(Attempt1, self).__init__(dims)
        self.name = "Attempt1_3D"

    def build(self):
        frames = self.input.shape[1]
        mid_frame = int(frames / 2)
        width = self.input.shape[3]
        height = self.input.shape[2]
        channels = self.input.shape[4]
        conv1 = Conv3D(filters=64, kernel_size=(1, 3, 3), activation="relu")(self.input)
        zpad1 = ZeroPadding3D(padding=(0, 2, 2))(conv1)
        conv2 = Conv3D(filters=32, kernel_size=(1, 3, 3), activation="relu")(zpad1)
        mpool1 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
        conv3 = Conv3D(filters=64, kernel_size=(1, 3, 3), activation="tanh")(mpool1)
        drop1 = Dropout(0.05)(conv3)
        encode = Conv3D(filters=3, kernel_size=(1, 2, 2), strides=(1, 2, 2))(drop1)

        # model.add(Flatten())
        up1 = UpSampling3D(size=(1, 2, 2))(encode)
        conv4 = Conv3DTranspose(
            filters=8, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="valid"
        )(up1)
        conv5 = Conv3DTranspose(
            filters=16,
            kernel_size=(1, 3, 3),
            strides=(1, 1, 1),
            padding="valid",
            kernel_regularizer=regularizers.l2(0.01),
        )(conv4)
        up2 = UpSampling3D(size=(1, 2, 2))(conv5)
        conv6 = Conv3DTranspose(
            filters=32, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same"
        )(up2)
        zpad2 = ZeroPadding3D(padding=(0, 2, 2))(conv6)
        drop2 = GaussianDropout(0.02)(zpad2)
        conv7 = Conv3DTranspose(
            filters=16, kernel_size=(1, 3, 3), activation="relu", padding="same"
        )(drop2)
        conv8 = Conv3DTranspose(
            filters=8, kernel_size=(1, 5, 5), strides=(1, 1, 1), padding="same"
        )(conv7)
        conv9 = Conv3D(
            filters=3, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="valid"
        )(conv8)
        decode = self.crop(1, mid_frame, mid_frame + 1)(conv9)

        model = Model(self.input, decode)

        model.name = self.name

        model.c_space = self.c_space

        return model


class UNet_3D(ModelClass):
    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(dims, precision, **kwargs)
        self.name = "U-Net 3D"

    def build(self):
        conv1 = Conv3D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(self.input)
        conv1 = Conv3D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv1)
        pool1 = MaxPooling3D(pool_size=(1, 2, 2))(conv1)
        conv2 = Conv3D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool1)
        conv2 = Conv3D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv2)
        pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
        conv3 = Conv3D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool2)
        conv3 = Conv3D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv3)
        pool3 = MaxPooling3D(pool_size=(1, 2, 2))(conv3)
        conv4 = Conv3D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool3)
        conv4 = Conv3D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling3D(pool_size=(1, 2, 2))(drop4)

        conv5 = Conv3D(
            1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(pool4)
        conv5 = Conv3D(
            1024, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv3D(
            512, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling3D(size=(1, 2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=4)
        # merge6 = Concatenate([drop4, up6], axis=3)
        conv6 = Conv3D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge6)
        conv6 = Conv3D(
            512, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv6)

        up7 = Conv3D(
            256, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling3D(size=(1, 2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=4)
        conv7 = Conv3D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge7)
        conv7 = Conv3D(
            256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv7)

        up8 = Conv3D(
            128, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling3D(size=(1, 2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=4)
        conv8 = Conv3D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge8)
        conv8 = Conv3D(
            128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv8)

        up9 = Conv3D(
            64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        )(UpSampling3D(size=(1, 2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=4)
        conv9 = Conv3D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(merge9)
        conv9 = Conv3D(
            64, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv9)
        decode = Conv3D(
            3, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        )(conv9)

        model = Model(self.input, decode)

        model.name = self.name

        model.c_space = self.c_space

        return model


class Attempt5(ModelClass):
    """
    Similar architecture to Attempt1_3D, but using some skip connections, also is symmetric
    """

    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(
            dims, precision, **kwargs
        )  # Is equivalent to super(Attempt1, self).__init__(dims)
        self.name = "Attempt5"

    def build(self):
        frames = self.input.shape[1]
        mid_frame = int(frames / 2)
        width = self.input.shape[3]
        height = self.input.shape[2]
        channels = self.input.shape[4]
        # conv2 = Conv3D(
        #     128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        # )(conv2)
        # pool2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
        # conv7 = Conv3D(
        #     256, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        # )(conv7)
        #
        # up8 = Conv3D(
        #     128, 2, activation="relu", padding="same", kernel_initializer="he_normal"
        # )(UpSampling3D(size=(1, 2, 2))(conv7))
        # merge8 = concatenate([conv2, up8], axis=4)
        # conv8 = Conv3D(
        #     128, 3, activation="relu", padding="same", kernel_initializer="he_normal"
        # )(merge8)
        # single_frame = (1,) + self.input.shape[2:]
        conv1 = Conv3D(filters=64, kernel_size=(1, 3, 3), activation="relu")(self.input)
        zpad1 = ZeroPadding3D(padding=(0, 2, 2))(conv1)
        conv2 = Conv3D(filters=32, kernel_size=(1, 3, 3), activation="relu")(zpad1)
        maxpool1 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
        conv3 = Conv3D(filters=64, kernel_size=(1, 3, 3), activation="tanh")(maxpool1)
        drop1 = Dropout(0.05)(conv3)
        conv4 = Conv3D(filters=3, kernel_size=(1, 2, 2), strides=(1, 2, 2))(drop1)

        # Begin decoding
        up1 = UpSampling3D(size=(1, 2, 2))(conv4)
        conv5 = Conv3DTranspose(
            filters=8, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="valid"
        )(up1)
        conv6 = Conv3DTranspose(
            filters=16,
            kernel_size=(1, 3, 3),
            strides=(1, 1, 1),
            padding="valid",
            kernel_regularizer=regularizers.l2(0.01),
        )(conv5)
        up2 = UpSampling3D(size=(1, 2, 2))(conv6)
        conv8 = Conv3DTranspose(
            filters=32, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same"
        )(up2)
        zpad2 = ZeroPadding3D(padding=(0, 2, 2))(conv8)
        drop2 = GaussianDropout(0.02)(zpad2)
        conv9 = Conv3DTranspose(
            filters=16, kernel_size=(1, 3, 3), activation="relu", padding="same"
        )(drop2)
        conv10 = Conv3DTranspose(
            filters=8, kernel_size=(1, 5, 5), strides=(1, 1, 1), padding="same"
        )(conv9)
        # Tensor is (batch_size, frames, height, width, channels)
        # x = Reshape(single_frame)(x)  # Convert to single frame
        # perm1 = Permute((2, 3, 4))(conv10)
        conv11 = Conv3D(
            filters=3, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="valid"
        )(conv10)
        decode = self.crop(1, mid_frame, mid_frame + 1)(conv11)

        model = Model(self.input, decode)

        model.name = self.name

        model.c_space = self.c_space

        return model


class LSTM1(ModelClass):
    """
    Attempt1 (Attempt1_3D) architecture applied to LSTMs
    """

    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(
            dims, precision, **kwargs
        )  # Is equivalent to super(Attempt1, self).__init__(dims)
        self.name = "LSTM1"

    def build(self):
        frames = self.input.shape[1]
        mid_frame = int(frames / 2)
        width = self.input.shape[3]
        height = self.input.shape[2]
        channels = self.input.shape[4]

        conv1 = ConvLSTM2D(
            filters=64, kernel_size=(3, 3), activation="relu", return_sequences=True
        )(self.input)
        zpad1 = ZeroPadding3D(padding=(0, 2, 2))(conv1)
        conv2 = ConvLSTM2D(
            filters=32, kernel_size=(3, 3), activation="relu", return_sequences=True
        )(zpad1)
        mpool1 = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
        conv3 = ConvLSTM2D(
            filters=64, kernel_size=(3, 3), activation="tanh", return_sequences=True
        )(mpool1)
        drop1 = Dropout(0.05)(conv3)
        conv4 = ConvLSTM2D(
            filters=8, kernel_size=(2, 2), strides=(2, 2), return_sequences=True
        )(drop1)
        # Get first 3 frames
        preceeding_frames = self.crop(1, 0, mid_frame + 1)(conv4)
        # preceeding_frames_1 = Lambda(lambda x: conv4[:, : (mid_frame + 1)])(conv4)
        # preceeding_frames_2 = Cropping3D(cropping=((0, mid_frame), (0, 0), (0, 0)))(conv4)
        # Produce 'new' 3rd frame
        encode = ConvLSTM2D(
            filters=channels, kernel_size=(1, 1), return_sequences=False
        )(preceeding_frames)

        up1 = UpSampling2D(size=(2, 2))(encode)
        conv5 = Conv2DTranspose(
            filters=8, kernel_size=2, strides=(2, 2), padding="valid"
        )(up1)
        conv6 = Conv2DTranspose(
            filters=16,
            kernel_size=3,
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=regularizers.l2(0.01),
        )(conv5)
        up2 = UpSampling2D(size=(2, 2))(conv6)
        conv7 = Conv2DTranspose(
            filters=32, kernel_size=3, strides=(1, 1), padding="same"
        )(up2)
        zpad2 = ZeroPadding2D(padding=(2, 2))(conv7)
        drop2 = GaussianDropout(0.02)(zpad2)
        conv8 = Conv2DTranspose(
            filters=16, kernel_size=3, activation="relu", padding="same"
        )(drop2)
        conv9 = Conv2DTranspose(
            filters=8, kernel_size=5, strides=(1, 1), padding="same"
        )(conv8)
        conv10 = Conv2D(
            filters=channels, kernel_size=2, strides=(2, 2), padding="valid"
        )(conv9)
        # To get the output to agree with ndims
        decode = Reshape(target_shape=(1, height, width, channels))(conv10)
        # up1 = UpSampling2D(size=(2, 2))(encode)
        # conv5 = Conv3DTranspose(
        #     filters=8, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="valid"
        # )(up1)
        # conv6 = Conv3DTranspose(
        #     filters=16,
        #     kernel_size=(1, 3, 3),
        #     strides=(1, 1, 1),
        #     padding="valid",
        #     kernel_regularizer=regularizers.l2(0.01),
        # )(conv5)
        # up2 = UpSampling3D(size=(1, 2, 2))(conv6)
        # conv7 = Conv3DTranspose(
        #     filters=32, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same"
        # )(up2)
        # zpad2 = ZeroPadding3D(padding=(0, 2, 2))(conv7)
        # drop2 = GaussianDropout(0.02)(zpad2)
        # conv8 = Conv3DTranspose(
        #     filters=16, kernel_size=(1, 3, 3), activation="relu", padding="same"
        # )(drop2)
        # conv9 = Conv3DTranspose(
        #     filters=8, kernel_size=(1, 5, 5), strides=(1, 1, 1), padding="same"
        # )(conv8)
        # decode = Conv3D(
        #     filters=3, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="valid"
        # )(conv9)

        model = Model(self.input, decode)

        model.name = self.name

        model.c_space = self.c_space

        return model


class LSTM2(ModelClass):
    """
    Attempt1 (Attempt1_3D) architecture applied to LSTMs, using forward and backward prediction
    """

    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(
            dims, precision, **kwargs
        )  # Is equivalent to super(Attempt1, self).__init__(dims)
        self.name = "LSTM2"

    def build(self):
        frames = self.input.shape[1]
        mid_frame = int(frames / 2)
        width = self.input.shape[3]
        height = self.input.shape[2]
        channels = self.input.shape[4]

        # Input layer
        conv1 = ConvLSTM2D(
            filters=64, kernel_size=(3, 3), activation="relu", return_sequences=True
        )(self.input)
        zpad1 = ZeroPadding3D(padding=(0, 2, 2))(conv1)
        # Forward
        forward_frames = self.crop(1, 0, mid_frame + 1)(zpad1)
        conv2_1 = ConvLSTM2D(
            filters=32, kernel_size=(3, 3), activation="relu", return_sequences=True
        )(forward_frames)
        mpool1_1 = MaxPooling3D(pool_size=(1, 2, 2))(conv2_1)
        conv3_1 = ConvLSTM2D(
            filters=64, kernel_size=(3, 3), activation="tanh", return_sequences=True
        )(mpool1_1)
        drop1_1 = Dropout(0.05)(conv3_1)
        conv4_1 = ConvLSTM2D(
            filters=8, kernel_size=(2, 2), strides=(2, 2), return_sequences=False
        )(drop1_1)
        # target_shape = (1, conv4_1.shape[1], conv4_1.shape[2], conv4_1.shape[3])
        # reshape_1 = Reshape(target_shape=target_shape)(conv4_1)
        # Backward
        backward_frames = self.crop(1, mid_frame, -1)(zpad1)
        conv2_2 = ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            return_sequences=True,
            go_backwards=True,
        )(backward_frames)
        mpool1_2 = MaxPooling3D(pool_size=(1, 2, 2))(conv2_2)
        conv3_2 = ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            activation="tanh",
            return_sequences=True,
            go_backwards=True,
        )(mpool1_2)
        drop1_2 = Dropout(0.05)(conv3_2)
        conv4_2 = ConvLSTM2D(
            filters=8,
            kernel_size=(2, 2),
            strides=(2, 2),
            return_sequences=False,
            go_backwards=True,
        )(drop1_2)
        # target_shape = (1, conv4_2.shape[1], conv4_2.shape[2], conv4_2.shape[3])
        # reshape_2 = Reshape(target_shape=target_shape)(conv4_1)
        # Get the 3rd frame
        merge1 = concatenate([conv4_1, conv4_2], axis=3)
        # merge1 = concatenate([reshape_1, reshape_2], axis=4)
        encode = Conv2D(filters=channels, kernel_size=(1, 1))(merge1)

        up1 = UpSampling2D(size=(2, 2))(encode)
        conv5 = Conv2DTranspose(
            filters=8, kernel_size=2, strides=(2, 2), padding="valid"
        )(up1)
        conv6 = Conv2DTranspose(
            filters=16,
            kernel_size=3,
            strides=(1, 1),
            padding="valid",
            kernel_regularizer=regularizers.l2(0.01),
        )(conv5)
        up2 = UpSampling2D(size=(2, 2))(conv6)
        conv7 = Conv2DTranspose(
            filters=32, kernel_size=3, strides=(1, 1), padding="same"
        )(up2)
        zpad2 = ZeroPadding2D(padding=(2, 2))(conv7)
        drop2 = GaussianDropout(0.02)(zpad2)
        conv8 = Conv2DTranspose(
            filters=16, kernel_size=3, activation="relu", padding="same"
        )(drop2)
        conv9 = Conv2DTranspose(
            filters=8, kernel_size=5, strides=(1, 1), padding="same"
        )(conv8)
        conv10 = Conv2D(
            filters=channels, kernel_size=2, strides=(2, 2), padding="valid"
        )(conv9)
        # To get the output to agree with ndims
        decode = Reshape(target_shape=(1, height, width, channels))(conv10)
        # conv5 = Conv3DTranspose(
        #     filters=8, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="valid"
        # )(up1)
        # conv6 = Conv3DTranspose(
        #     filters=16,
        #     kernel_size=(1, 3, 3),
        #     strides=(1, 1, 1),
        #     padding="valid",
        #     kernel_regularizer=regularizers.l2(0.01),
        # )(conv5)
        # up2 = UpSampling3D(size=(1, 2, 2))(conv6)
        # conv7 = Conv3DTranspose(
        #     filters=32, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same"
        # )(up2)
        # zpad2 = ZeroPadding3D(padding=(0, 2, 2))(conv7)
        # drop2 = GaussianDropout(0.02)(zpad2)
        # conv8 = Conv3DTranspose(
        #     filters=16, kernel_size=(1, 3, 3), activation="relu", padding="same"
        # )(drop2)
        # conv9 = Conv3DTranspose(
        #     filters=8, kernel_size=(1, 5, 5), strides=(1, 1, 1), padding="same"
        # )(conv8)
        # decode = Conv3D(
        #     filters=3, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="valid"
        # )(conv9)

        model = Model(self.input, decode)

        model.name = self.name

        model.c_space = self.c_space

        return model


# Conv2D(filters=z, kernel_size=1, strides=(1, 1), padding="same") -> (x, y, z)
# Conv2D(filters=z, kernel_size=1, strides=(1, 1), padding="valid")-> (x, y, z)
# Conv2D(filters=z, kernel_size=1, strides=(2, 2), padding="same") -> (x/2, y/2, z) [ceil]
# Conv2D(filters=z, kernel_size=1, strides=(2, 2), padding="valid") -> (x/2, y/2, z) [ceil ?]
# Conv2D(filters=z, kernel_size=2, strides=(1, 1), padding="same") -> (x, y, z)
# Conv2D(filters=z, kernel_size=2, strides=(1, 1), padding="valid")-> (x-1, y-1, z)
# Conv2D(filters=z, kernel_size=2, strides=(2, 2), padding="same") -> (x/2, y/2, z) [ceil]
# Conv2D(filters=z, kernel_size=2, strides=(2, 2), padding="valid") -> ((x-1)/2, (y-1)/2, z) [ceil]
# Conv2D(filters=z, kernel_size=3, strides=(1, 1), padding="same") -> (x, y, z)
# Conv2D(filters=z, kernel_size=3, strides=(1, 1), padding="valid") -> (x-2, y-2, z)
# Conv2D(filters=z, kernel_size=3, strides=(2, 2), padding="same") -> (x/2, y/2, z) [ceil]
# Conv2D(filters=z, kernel_size=3, strides=(2, 2), padding="valid") -> ((x-2)/2, (y-2)/2, z) [ceil]
# Conv2D(filters=z, kernel_size=4, strides=(1, 1), padding="same") -> (x, y, z)
# Conv2D(filters=z, kernel_size=4, strides=(1, 1), padding="valid") -> (x-3, y-3, z)
# Conv2D(filters=z, kernel_size=4, strides=(2, 2), padding="same") -> (x/2, y/2, z) [ceil]
# Conv2D(filters=z, kernel_size=4, strides=(2, 2), padding="valid") -> ((x-3)/2, (y-3)/2, z) [ceil]
