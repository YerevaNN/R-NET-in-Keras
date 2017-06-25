import numpy as np
from keras import backend as K
from keras.engine import Layer, InputSpec

class Argmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Argmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        P = K.shape(inputs) [-1] // 2
        start = K.argmax(inputs[:, :P])
        end = K.argmax(inputs[:, P:])
        return [start, end]

    def compute_output_shape(self, input_shape):
        return [input_shape[:-1], input_shape[:-1]]

    def compute_mask(self, x, mask):
        return [None, None]
