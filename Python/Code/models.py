"""
File for all models, to be used by main.py
"""
import os
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
    Cropping3D,
    add,
    Flatten,
    BatchNormalization,
    Activation,
    Dense,
)
from keras import regularizers
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    ReduceLROnPlateau,
)
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


class ModelClass:
    def __init__(self, dims, precision="float32", **kwargs):
        self.set_precision(precision)
        self.input = self.input_layer(dims)

    @staticmethod
    def input_layer(dims):
        return Input(shape=dims, name="input")

    @staticmethod
    def set_precision(precision="float32"):
        K.set_floatx(precision)

    @staticmethod
    def ready_training(
        compressed_images, original_images, split=0.2, state=42, **kwargs
    ):
        # test_list_expanded = np.expand_dims(test_list, axis=0)
        # test_list_other = np.expand_dims(test_list, axis=1)

        # train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        # test_datagen = ImageDataGenerator(rescale=1./255)
        #
        # training_images = train_datagen.flow_from_directory(compressed_images_path, target_size=(512, 768),
        #                                                     color_mode='rgb', batch_size=32)
        #
        # testing_images = test_datagen.flow_from_directory(compressed_images_path, target_size=(512, 768),
        #                                                   color_mode='rgb', batch_size=32)

        # X_train, X_val, y_train, y_val = train_test_split(training_images, testing_images,
        #                                                   test_size=0.2, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(
            compressed_images, original_images, test_size=split, random_state=state
        )

        return x_train, x_val, y_train, y_val

    def train(
        self,
        model,
        train=None,
        label=None,
        x_train=None,
        x_val=None,
        y_train=None,
        y_val=None,
        generator=False,
        run_epochs=10,
        batch_size=2,
        util_class=None,
        **kwargs,
    ):
        if not generator and (
            x_train is None or x_val is None or y_train is None or y_val is None
        ):
            x_train, x_val, y_train, y_val = self.ready_training(train, label, **kwargs)
        cb_patience = int(run_epochs * 0.15)
        cb = [
            EarlyStopping(
                verbose=True, patience=cb_patience, monitor="val_tf_ssim", mode="min"
            )
        ]
        # "val_tf_psnr"
        # "val_loss" (MS-SSIM)
        print("Training")
        start = timer()

        if generator:
            # TODO - Alter max_queue_size ?
            history = model.fit_generator(
                util_class.generator_function(batch_size=batch_size, **kwargs),
                steps_per_epoch=144 // batch_size,  # Hardcoded --> 144 images in /Test
                epochs=run_epochs,
                verbose=2,
                callbacks=cb,
                validation_data=None,
                validation_steps=None,
                validation_freq=1,
                class_weight=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                shuffle=True,
                initial_epoch=0,
            )
        else:
            history = model.fit(
                x_train,
                y_train,
                epochs=run_epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
                shuffle=True,
                verbose=2,
                callbacks=cb,
            )

        # TODO record metrics from screen on best iteration, record metrics for early stop and time to train

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

        return model


class Attempt2(ModelClass):
    """
    Model seems too large current, cannot allocate layer (2048, 3072, 69)
    """

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
        # TODO - Upscale more
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

        return model


class Attempt3(ModelClass):
    """
    Model seems too large current, cannot allocate layer (2048, 3072, 69)
    """

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

        return model


class Attempt1_3D(ModelClass):
    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(
            dims, precision, **kwargs
        )  # Is equivalent to super(Attempt1, self).__init__(dims)
        self.name = "Attempt1_3D"

    def build(self):
        x = Conv3D(filters=64, kernel_size=(1, 3, 3), activation="relu")(self.input)
        x = ZeroPadding3D(padding=(0, 2, 2))(x)
        x = Conv3D(filters=32, kernel_size=(1, 3, 3), activation="relu")(x)
        x = MaxPooling3D(pool_size=(1, 2, 2))(x)
        x = Conv3D(filters=64, kernel_size=(1, 3, 3), activation="tanh")(x)
        x = Dropout(0.05)(x)
        encode = Conv3D(filters=3, kernel_size=(1, 2, 2), strides=(1, 2, 2))(x)

        # model.add(Flatten())
        x = UpSampling3D(size=(1, 2, 2))(encode)
        x = Conv3DTranspose(
            filters=8, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="valid"
        )(x)
        x = Conv3DTranspose(
            filters=16,
            kernel_size=(1, 3, 3),
            strides=(1, 1, 1),
            padding="valid",
            kernel_regularizer=regularizers.l2(0.01),
        )(x)
        x = UpSampling3D(size=(1, 2, 2))(x)
        x = Conv3DTranspose(
            filters=32, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same"
        )(x)
        x = ZeroPadding3D(padding=(0, 2, 2))(x)
        x = GaussianDropout(0.02)(x)
        x = Conv3DTranspose(
            filters=16, kernel_size=(1, 3, 3), activation="relu", padding="same"
        )(x)
        x = Conv3DTranspose(
            filters=8, kernel_size=(1, 5, 5), strides=(1, 1, 1), padding="same"
        )(x)
        decode = Conv3D(
            filters=3, kernel_size=(1, 2, 2), strides=(1, 2, 2), padding="valid"
        )(x)

        model = Model(self.input, decode)

        model.name = self.name

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


class ResNet(ModelClass):
    def __init__(self, dims, precision="float32", **kwargs):
        super().__init__(
            dims, precision, **kwargs
        )  # Is equivalent to super(KerasDenoise, self).__init__(dims)
        self.name = "ResNet"

    def build(self, version=2, n=3):
        if version == 2:
            depth = n * 9 + 2
            model = self.resnet_v2(depth=depth)
        else:
            depth = n * 6 + 2
            model = self.resnet_v1(depth=depth)

        model.name = self.name + f"{depth}_v{version}"

        return model

    @staticmethod
    def resnet_layer(
        inputs,
        num_filters=16,
        kernel_size=3,
        strides=1,
        activation="relu",
        batch_normalization=True,
        conv_first=True,
    ):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)

        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(
            num_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-4),
        )

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def resnet_v1(self, depth, num_classes=10):
        """ResNet Version 1 Model builder [a]

        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        # inputs = Input(shape=self.input)
        x = self.resnet_layer(inputs=self.input)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = self.resnet_layer(
                    inputs=x, num_filters=num_filters, strides=strides
                )
                y = self.resnet_layer(
                    inputs=y, num_filters=num_filters, activation=None
                )
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(
                        inputs=x,
                        num_filters=num_filters,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normalization=False,
                    )
                x = add([x, y])
                x = Activation("relu")(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        outputs = AveragePooling2D(pool_size=8)(x)
        # Match dimensions for images
        # y = Flatten()(x)
        # outputs = Dense(
        #     num_classes, activation="softmax", kernel_initializer="he_normal"
        # )(y)

        # Instantiate model.
        model = Model(inputs=self.input, outputs=outputs)
        return model

    def resnet_v2(self, depth, num_classes=10):
        """ResNet Version 2 Model builder [b]

        Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
        bottleneck layer
        First shortcut connection per layer is 1 x 1 Conv2D.
        Second and onwards shortcut connection is identity.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filter maps is
        doubled. Within each stage, the layers have the same number filters and the
        same filter map sizes.
        Features maps sizes:
        conv1  : 32x32,  16
        stage 0: 32x32,  64
        stage 1: 16x16, 128
        stage 2:  8x8,  256

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 9 != 0:
            raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")
        # Start model definition.
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        # inputs = Input(shape=self.input) # Is self.input
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = self.resnet_layer(
            inputs=self.input, num_filters=num_filters_in, conv_first=True
        )

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = "relu"
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2  # downsample

                # bottleneck residual unit
                y = self.resnet_layer(
                    inputs=x,
                    num_filters=num_filters_in,
                    kernel_size=1,
                    strides=strides,
                    activation=activation,
                    batch_normalization=batch_normalization,
                    conv_first=False,
                )
                y = self.resnet_layer(
                    inputs=y, num_filters=num_filters_in, conv_first=False
                )
                y = self.resnet_layer(
                    inputs=y,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    conv_first=False,
                )
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(
                        inputs=x,
                        num_filters=num_filters_out,
                        kernel_size=1,
                        strides=strides,
                        activation=None,
                        batch_normalization=False,
                    )
                x = add([x, y])

            num_filters_in = num_filters_out

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        outputs = AveragePooling2D(pool_size=8)(x)
        # Match dimensions
        # y = Flatten()(x)
        # outputs = Dense(
        #     num_classes, activation="softmax", kernel_initializer="he_normal"
        # )(y)

        # Instantiate model.
        model = Model(inputs=self.input, outputs=outputs)
        return model

    def train(
        self,
        model,
        train=None,
        label=None,
        x_train=None,
        x_val=None,
        y_train=None,
        y_val=None,
        subtract_pixel_mean=False,
        data_augmentation=False,
        generator=False,
        **kwargs,
    ):
        if not generator and (
            x_train is None or x_val is None or y_train is None or y_val is None
        ):
            x_train, x_val, y_train, y_val = self.ready_training(train, label, **kwargs)

        run_epochs = 200
        batch_size = 32  # orig paper trained all networks with batch_size=128

        # Subtracting pixel mean improves accuracy
        if subtract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            x_train -= x_train_mean
            x_val -= x_train_mean

        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_val.shape[0], "test samples")
        print("y_train shape:", y_train.shape)

        # Handled outside currently
        # model.compile(
        #     loss="categorical_crossentropy",
        #     optimizer=Adam(learning_rate=self.lr_schedule(0)),
        #     metrics=["accuracy"],
        # )
        # model.summary()

        # Prepare callbacks for model saving and for learning rate adjustment.
        cb_patience = int(run_epochs * 0.15)
        # checkpoint = ModelCheckpoint(
        #     filepath=filepath, monitor="val_acc", verbose=1, save_best_only=True
        # )

        lr_scheduler = LearningRateScheduler(self.lr_schedule)

        lr_reducer = ReduceLROnPlateau(
            factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6
        )

        early_stop = EarlyStopping(
            verbose=True, patience=cb_patience, monitor="val_tf_ssim", mode="min"
        )

        # callbacks = [checkpoint, lr_reducer, lr_scheduler]

        cb = [lr_reducer, lr_scheduler, early_stop]

        # "val_tf_psnr"
        # "val_loss" (MS-SSIM)
        print("Training")
        start = timer()

        # history = model.fit(
        #     x_train,
        #     y_train,
        #     epochs=run_epochs,
        #     batch_size=batch_size,
        #     validation_data=(x_val, y_val),
        #     shuffle=True,
        #     verbose=2,
        #     callbacks=cb,
        # )

        # Run training, with or without data augmentation.
        if not data_augmentation:
            print("Not using data augmentation.")
            history = model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=run_epochs,
                validation_data=(x_val, y_val),
                shuffle=True,
                callbacks=cb,
            )
        else:
            print("Using real-time data augmentation.")
            # This will do pre-processing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.0,
                # set range for random zoom
                zoom_range=0.0,
                # set range for random channel shifts
                channel_shift_range=0.0,
                # set mode for filling points outside the input boundaries
                fill_mode="nearest",
                # value used for fill_mode = "constant"
                cval=0.0,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0,
            )

            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            history = model.fit_generator(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                validation_data=(x_val, y_val),
                epochs=run_epochs,
                verbose=1,
                workers=4,
                callbacks=cb,
            )

        # TODO record metrics from screen on best iteration, record metrics for early stop and time to train

        end = timer()
        dur = end - start
        print("Training took: {}".format(str(datetime.timedelta(seconds=dur))))

        return history

    # Training parameters
    # batch_size = 32  # orig paper trained all networks with batch_size=128
    # epochs = 200
    # data_augmentation = True
    # num_classes = 10

    # # Subtracting pixel mean improves accuracy
    # subtract_pixel_mean = True

    # Model parameter
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
    #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
    # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
    # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
    # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
    # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
    # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
    # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
    # ---------------------------------------------------------------------------
    # n = 3

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    # version = 1

    # Computed depth from supplied model parameter n
    # if version == 1:
    #     depth = n * 6 + 2
    # elif version == 2:
    #     depth = n * 9 + 2

    # Model name, depth and version
    # model_type = "ResNet%dv%d" % (depth, version)

    # Load the CIFAR10 data.
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # (x_train, y_train), (x_test, y_test) = (None, None), (None, None)

    # Input image dimensions.
    # input_shape = x_train.shape[1:]

    # Normalize data.
    # x_train = x_train.astype("float32") / 255
    # x_test = x_test.astype("float32") / 255

    # If subtract pixel mean is enabled
    # if subtract_pixel_mean:
    #     x_train_mean = np.mean(x_train, axis=0)
    #     x_train -= x_train_mean
    #     x_test -= x_train_mean

    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")
    # print("y_train shape:", y_train.shape)

    # Convert class vectors to binary class matrices.
    # y_train = None  # keras.utils.to_categorical(y_train, num_classes)
    # y_test = None  # keras.utils.to_categorical(y_test, num_classes)

    # model = None
    # model.compile(
    #     loss="categorical_crossentropy",
    #     optimizer=Adam(learning_rate=self.lr_schedule(0)),
    #     metrics=["accuracy"],
    # )
    # model.summary()
    # print(model_type)

    # Prepare model model saving directory.
    # save_dir = os.path.join(os.getcwd(), "saved_models")
    # model_name = "cifar10_%s_model.{epoch:03d}.h5" % model_type
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # filepath = os.path.join(save_dir, model_name)

    # # Prepare callbacks for model saving and for learning rate adjustment.
    # checkpoint = ModelCheckpoint(
    #     filepath=filepath, monitor="val_acc", verbose=1, save_best_only=True
    # )
    #
    # lr_scheduler = LearningRateScheduler(lr_schedule)
    #
    # lr_reducer = ReduceLROnPlateau(
    #     factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6
    # )
    #
    # callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # # Run training, with or without data augmentation.
    # if not data_augmentation:
    #     print("Not using data augmentation.")
    #     model.fit(
    #         x_train,
    #         y_train,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         validation_data=(x_test, y_test),
    #         shuffle=True,
    #         callbacks=callbacks,
    #     )
    # else:
    #     print("Using real-time data augmentation.")
    #     # This will do preprocessing and realtime data augmentation:
    #     datagen = ImageDataGenerator(
    #         # set input mean to 0 over the dataset
    #         featurewise_center=False,
    #         # set each sample mean to 0
    #         samplewise_center=False,
    #         # divide inputs by std of dataset
    #         featurewise_std_normalization=False,
    #         # divide each input by its std
    #         samplewise_std_normalization=False,
    #         # apply ZCA whitening
    #         zca_whitening=False,
    #         # epsilon for ZCA whitening
    #         zca_epsilon=1e-06,
    #         # randomly rotate images in the range (deg 0 to 180)
    #         rotation_range=0,
    #         # randomly shift images horizontally
    #         width_shift_range=0.1,
    #         # randomly shift images vertically
    #         height_shift_range=0.1,
    #         # set range for random shear
    #         shear_range=0.0,
    #         # set range for random zoom
    #         zoom_range=0.0,
    #         # set range for random channel shifts
    #         channel_shift_range=0.0,
    #         # set mode for filling points outside the input boundaries
    #         fill_mode="nearest",
    #         # value used for fill_mode = "constant"
    #         cval=0.0,
    #         # randomly flip images
    #         horizontal_flip=True,
    #         # randomly flip images
    #         vertical_flip=False,
    #         # set rescaling factor (applied before any other transformation)
    #         rescale=None,
    #         # set function that will be applied on each input
    #         preprocessing_function=None,
    #         # image data format, either "channels_first" or "channels_last"
    #         data_format=None,
    #         # fraction of images reserved for validation (strictly between 0 and 1)
    #         validation_split=0.0,
    #     )
    #
    #     # Compute quantities required for featurewise normalization
    #     # (std, mean, and principal components if ZCA whitening is applied).
    #     datagen.fit(x_train)
    #
    #     # Fit the model on the batches generated by datagen.flow().
    #     model.fit_generator(
    #         datagen.flow(x_train, y_train, batch_size=batch_size),
    #         validation_data=(x_test, y_test),
    #         epochs=epochs,
    #         verbose=1,
    #         workers=4,
    #         callbacks=callbacks,
    #     )

    # Score trained model.
    # scores = model.evaluate(x_test, y_test, verbose=1)
    # print("Test loss:", scores[0])
    # print("Test accuracy:", scores[1])
