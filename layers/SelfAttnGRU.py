# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K

from WrappedGRU import WrappedGRU
from helpers import compute_mask, softmax

class SelfAttnGRU(WrappedGRU):

    def build(self, input_shape):
        H = self.units
        assert(isinstance(input_shape, list))
        
        nb_inputs = len(input_shape)
        assert(nb_inputs >= 2)

        assert(len(input_shape[0]) == 3)
        B, P, H_ = input_shape[0]
        assert(H_ == H)


        assert(len(input_shape[1]) == 3)
        B, P_, H_ = input_shape[1]
        assert(P_ == P)
        assert(H_ == H)

        self.input_spec = [None]
        super(SelfAttnGRU, self).build(input_shape=(B, P, 2 * H))
        self.GRU_input_spec = self.input_spec
        self.input_spec = [None] * nb_inputs

    def step(self, inputs, states):
        vP_t = inputs
        hP_tm1 = states[0]
        _ = states[1:3] # ignore internal dropout/masks 
        vP, WP_v, WPP_v, v, W_g2 = states[3:8]
        vP_mask, = states[8:]

        WP_v_Dot = K.dot(vP, WP_v)
        WPP_v_Dot = K.dot(K.expand_dims(vP_t, axis=1), WPP_v)

        s_t_hat = K.tanh(WPP_v_Dot + WP_v_Dot)
        s_t = K.dot(s_t_hat, v)
        s_t = K.batch_flatten(s_t)

        a_t = softmax(s_t, mask=vP_mask, axis=1)

        c_t = K.batch_dot(a_t, vP, axes=[1, 1])
        
        GRU_inputs = K.concatenate([vP_t, c_t])
        g = K.dot(GRU_inputs, W_g2)
        GRU_inputs = g * GRU_inputs
        
        hP_t, s = super(SelfAttnGRU, self).step(GRU_inputs, states)

        return hP_t, s
