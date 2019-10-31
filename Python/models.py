"""
File for all models, to be used by main.py
"""
import datetime
from timeit import default_timer as timer
from keras import Model, Input
from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dropout,
    MaxPooling2D,
    AveragePooling2D,
    UpSampling2D,
    ZeroPadding2D,
    GaussianDropout,
)
from keras import regularizers
from sklearn.model_selection import train_test_split


class ModelClass:
    def __init__(self, dims):
        self.input = self.input_layer(dims)

    @staticmethod
    def input_layer(dims):
        return Input(shape=dims, name="input")

    @staticmethod
    def ready_training(compressed_images, original_images, split=0.2, state=42):
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
        train,
        label,
        x_train=None,
        x_val=None,
        y_train=None,
        y_val=None,
        **kwargs
    ):
        if x_train is None or x_val is None or y_train is None or y_val is None:
            x_train, x_val, y_train, y_val = self.ready_training(train, label, **kwargs)
        print("Training")
        start = timer()

        history = model.fit(
            x_train,
            y_train,
            epochs=1000,
            batch_size=4,
            validation_data=(x_val, y_val),
            shuffle=True,
            verbose=2,
        )

        end = timer()
        dur = end - start
        print("Training took: {}".format(str(datetime.timedelta(seconds=dur))))

        return history


class Attempt1(ModelClass):
    def __init__(self, dims):
        super().__init__(dims)  # Is equivalent to super(Attempt1, self).__init__(dims)
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

    def __init__(self, dims):
        super().__init__(dims)  # Is equivalent to super(Attempt2, self).__init__(dims)
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

    def __init__(self, dims):
        super().__init__(dims)  # Is equivalent to super(Attempt3, self).__init__(dims)
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


class KerasAE(ModelClass):
    def __init__(self, dims):
        super().__init__(dims)  # Is equivalent to super(KerasAE, self).__init__(dims)
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


class KerasDenoise(ModelClass):
    def __init__(self, dims):
        super().__init__(
            dims
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
