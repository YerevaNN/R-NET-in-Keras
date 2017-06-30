# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K
from keras.layers import Layer
from keras.layers.wrappers import TimeDistributed

from helpers import compute_mask, softmax

class QuestionPooling(Layer):

    def __init__(self, **kwargs):
        super(QuestionPooling, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        assert(isinstance(input_shape, list) and len(input_shape) == 4)

        input_shape = input_shape[0]
        B, Q, H = input_shape
        
        return (B, H)

    def build(self, input_shape):
        assert(isinstance(input_shape, list) and len(input_shape) == 4)
        input_shape = input_shape[0]
        
        B, Q, H_ = input_shape
        H = H_ // 2

    def call(self, inputs, mask=None):
        assert(isinstance(inputs, list) and len(inputs) == 4)
        uQ, WQ_u, WQ_v, v = inputs
        uQ_mask = mask[0] if mask is not None else None

        ones = K.ones_like(K.sum(uQ, axis=1, keepdims=True)) # (B, 1, 2H)
        s_hat = K.dot(uQ, WQ_u)
        s_hat += K.dot(ones, WQ_v)
        s_hat = K.tanh(s_hat)
        s = K.dot(s_hat, v)
        s = K.batch_flatten(s)

        a = softmax(s, mask=uQ_mask, axis=1)

        rQ = K.batch_dot(uQ, a, axes=[1, 1])

        return rQ

    def compute_mask(self, input, mask=None):
        return None
