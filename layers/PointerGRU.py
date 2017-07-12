# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K
from keras.layers import Layer
from keras.layers.wrappers import TimeDistributed

from WrappedGRU import WrappedGRU
from helpers import compute_mask, softmax

class PointerGRU(WrappedGRU):

    def build(self, input_shape):
        H = self.units // 2
        assert(isinstance(input_shape, list))

        nb_inputs = len(input_shape)
        assert(nb_inputs >= 6)

        assert(len(input_shape[0]) >= 2)
        B, T = input_shape[0][:2]


        assert(len(input_shape[1]) == 3)
        B, P, H_ = input_shape[1]
        assert(H_ == 2 * H)

        self.input_spec = [None]
        super(PointerGRU, self).build(input_shape=(B, T, 2 * H))
        self.GRU_input_spec = self.input_spec
        self.input_spec = [None] * nb_inputs # TODO TODO TODO

    def step(self, inputs, states):
        # input
        ha_tm1 = states[0] # (B, 2H)
        _ = states[1:3] # ignore internal dropout/masks
        hP, WP_h, Wa_h, v = states[3:7] # (B, P, 2H)
        hP_mask, = states[7:8]

        WP_h_Dot = K.dot(hP, WP_h) # (B, P, H)
        Wa_h_Dot = K.dot(K.expand_dims(ha_tm1, axis=1), Wa_h) # (B, 1, H)

        s_t_hat = K.tanh(WP_h_Dot + Wa_h_Dot) # (B, P, H)
        s_t = K.dot(s_t_hat, v) # (B, P, 1)
        s_t = K.batch_flatten(s_t) # (B, P)
        a_t = softmax(s_t, mask=hP_mask, axis=1) # (B, P)
        c_t = K.batch_dot(hP, a_t, axes=[1, 1]) # (B, 2H)

        GRU_inputs = c_t
        ha_t, (ha_t_,) = super(PointerGRU, self).step(GRU_inputs, states)
        
        return a_t, [ha_t]

    def compute_output_shape(self, input_shape):
        assert(isinstance(input_shape, list))

        nb_inputs = len(input_shape)
        assert(nb_inputs >= 5)

        assert(len(input_shape[0]) >= 2)
        B, T = input_shape[0][:2]

        assert(len(input_shape[1]) == 3)
        B, P, H_ = input_shape[1]

        if self.return_sequences:
            return (B, T, P)
        else:
            return (B, P)

    def compute_mask(self, inputs, mask=None):
        return None # TODO
