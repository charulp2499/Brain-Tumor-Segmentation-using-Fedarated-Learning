import os
import numpy as np
import tensorflow as tf
from glob import glob

def sum_scaled_weights(scaled_weight_list, weights):
    avg_grad = []
    for grad_list_tuple in zip(*scaled_weight_list):
        weighted_gradients = [tf.multiply(grad, weight) for grad, weight in zip(grad_list_tuple, weights)]
        layer_mean = tf.reduce_mean(weighted_gradients, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad