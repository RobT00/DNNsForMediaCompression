"""
File containing custom loss functions
"""
from keras import backend as K
import tensorflow as tf
import numpy as np

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
    return -(tf.image.psnr(y_pred, y_true, max_val))


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
