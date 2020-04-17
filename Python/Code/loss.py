"""
File containing custom loss functions
"""
from keras import backend as K
import tensorflow as tf
import numpy as np
import os
import tempfile
from keras_preprocessing.image import array_to_img
import shutil

dummy_loss_var = K.variable(0.0)


def dummy_loss(y_true, y_pred):
    """
    Dummy function as loss function stand-in, if needed
    :param y_true: Ground Truth
    :param y_pred: Predicted label
    :return: as defined globally for script, a dummy var
    """
    return dummy_loss_var


def psnr(y_true, y_pred):
    """
    Numpy implementation of Peak Signal to Noise Ratio
    :param y_true: Ground Truth
    :param y_pred: Predicted label
    :return: PSNR value
    """
    assert y_pred.shape == y_true.shape, (
        "Shapes must be the same to calculates PSNR:"
        "\ny_pred.shape: {}\ny_true.shape: {}".format(y_pred.shape, y_true.shape)
    )
    return -10.0 * np.log10(np.mean(np.square(y_pred - y_true)))


def tf_psnr(y_true, y_pred, max_val=1.0):
    """
    Tensorflow implementation of the Peak Signal to Noise Ratio
    :param y_true: Ground Truth
    :param y_pred: Predicted label
    :param max_val: Maximum value for pixel, 1.0 for scaled, 255 otherwise
    :return: PSNR, result is negated in order to minimise loss (maximise PSNR)
    """
    return -(tf.dtypes.cast(tf.image.psnr(y_pred, y_true, max_val), dtype=K.floatx()))


def tf_psnr_vid(y_true, y_pred, max_val=1.0):
    """
    Implementation of Peak Signal to Noise Ratio in a sequence
    :param y_true: Ground Truth sequence
    :param y_pred: Predicted sequence
    :param max_val: Maximum value for pixel, 1.0 for scaled, 255 otherwise
    :return: PSNR, result is negated in order to minimise loss (maximise PSNR)
    """
    if len(y_pred.shape) > 4:
        frames = y_pred.shape[1] if y_pred.shape[1] else 1
        mid_frame = int(frames / 2)
        pred_frame = y_pred[:, mid_frame, ...]
    else:
        pred_frame = y_pred
    # [batch_size, frames, height, width, channels]
    return tf_psnr(y_true, pred_frame, max_val=max_val)


def tf_ssim(y_true, y_pred, max_val=1.0):
    """
    Tensorflow implementation of Structural Similarity
    :param y_true: Ground Truth
    :param y_pred: Predicted label
    :param max_val: Maximum value for pixel, 1.0 for scaled, 255 otherwise
    :return: SSIM, result is negated in order to minimise loss (maximise SSIM)
    """
    return -(tf.dtypes.cast(tf.image.ssim(y_pred, y_true, max_val), dtype=K.floatx()))


def tf_ms_ssim(y_true, y_pred, max_val=1.0, filter_size=11):
    """
    Tensorflow implementation of Multi-Scale Structural Similarity
    :param y_true: Ground Truth
    :param y_pred: Predicted label
    :param max_val: Maximum value for pixel, 1.0 for scaled, 255 otherwise
    :param filter_size: Size of Gaussian filter
    :return: MS-SSIM, result is negated in order to minimise loss (maximise MS-SSIM)
    """
    return -(
        tf.dtypes.cast(
            tf.image.ssim_multiscale(y_pred, y_true, max_val, filter_size=filter_size),
            dtype=K.floatx(),
        )
    )


def tf_ms_ssim_vid(y_true, y_pred, max_val=1.0):
    """
    Implementation of Multi-Scale Structural Similarity for a frame in a sequence
    :param y_true: Ground Truth sequence
    :param y_pred: Predicted sequence
    :param max_val: Maximum value for pixel, 1.0 for scaled, 255 otherwise
    :return: MS-SSIM, result is negated in order to minimise loss (maximise MS-SIM)
    """
    if len(y_pred.shape) > 4:
        frames = y_pred.shape[1] if y_pred.shape[1] else 1
        mid_frame = int(frames / 2)
        pred_frame = y_pred[:, mid_frame, ...]
    else:
        pred_frame = y_pred
    # [batch_size, frames, height, width, channels]
    return tf_ms_ssim(y_true, pred_frame, max_val=max_val, filter_size=3)


def psnr_loss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    :param y_true: Ground Truth
    :param y_pred: Predicted label
    :return: PSNR value using Keras
    """
    return -10.0 * np.log10(K.mean(K.square(y_pred - y_true)))


def rmse(y_true, y_pred):
    """
    Loss function for Root Mean Squared Error
    :param y_true: Ground Truth
    :param y_pred: Predicted label
    :return: RMSE value
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def bytes_size(y_true, y_pred, rescale=255):
    """
    Save predicted image to disk to get file size, helping to minimise output file size
    :param y_true: Ground truth
    :param y_pred: Predicted image
    :return: Size of predicted image size (bytes)
    """
    s_dir = os.getcwd()
    tmp = tempfile.mkdtemp(dir=s_dir)
    os.chdir(tmp)
    # y_pred *= rescale
    # im = Image.fromarray(y_pred, mode=img_mode)
    im = array_to_img(y_pred)
    im.save("test.png")
    size = os.stat("test.png").st_size
    os.chdir(s_dir)
    shutil.rmtree(tmp)
    return size
