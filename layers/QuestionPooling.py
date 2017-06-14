# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K
from keras.layers import Layer
from keras.layers.wrappers import TimeDistributed

class QuestionPooling(Layer):

    def __init__(self, layer_map, layers=None, **kwargs):
        self.supports_masking = True
        super(QuestionPooling, self).__init__(**kwargs)

        self.layer_map = layer_map
        if layers is not None:
            self.set_layers(layers)

    @property
    def layers(self):
        return {k: self._layers[v] for k, v in self.layer_map.iteritems()}

    def set_layers(self, layers):
        self._layers = layers

    def get_config(self):
        config = {'layer_map': self.layer_map}
        base_config = super(WrappedGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            assert(len(input_shape) == 1)
            input_shape = input_shape[0]

        B, Q, H = input_shape
        
        return (B, H)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert(len(input_shape) == 1)
            input_shape = input_shape[0]
        
        B, Q, H_ = input_shape
        H = H_ // 2

        if not self.layers['e'].built:
            self.layers['e'].build(input_shape=(B, Q, 2 * H))
        
        if not self.layers['s'].built:
            self.layers['s'].build(input_shape=(B, 2 * H))
        
        if not self.layers['v'].built:
            self.layers['v'].build(input_shape=(B, Q, H))

    def call(self, inputs, mask=None):
        shape = K.shape(inputs)

        uQ = inputs
        ones = K.ones_like(K.sum(inputs, axis=1, keepdims=True))
        s_hat = self.layers['e'].call(uQ)
        s_hat += self.layers['s'].call(ones)
        s_hat = K.tanh(s_hat)
        s = self.layers['v'].call(s_hat)
        s = K.batch_flatten(s)
        a = K.softmax(s)
        rQ = K.batch_dot(uQ, a, axes=[1, 1])

        return rQ
