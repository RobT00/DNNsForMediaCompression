from keras import backend as K
import tensorflow as tf
import numpy as np

dummy_loss_var = K.variable(0.0)


def dummy_loss(y_true, y_pred):
    return dummy_loss_var


def psnr(y_true, y_pred):
    # assert y_pred.shape == y_true.shape, "Shapes must be the same to calculates PSNR:" \
    #                                      "\ny_pred.shape: {}\ny_true.shape: {}".format(y_pred.shape, y_true.shape)
    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))


def tf_psnr(y_true, y_pred, max_val=1.0):
    return -(tf.image.psnr(y_pred, y_true, max_val))


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * np.log10(K.mean(K.square(y_pred - y_true)))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
