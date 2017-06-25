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

        #
        #
        #

        assert(len(input_shape[-1]) == 3)
        B, P_, H_ = input_shape[-1]
        assert(P_ == P)
        assert(H_ == H)

        self.input_spec = [None]
        super(SelfAttnGRU, self).build(input_shape=(B, P, 2 * H))
        self.GRU_input_spec = self.input_spec
        self.input_spec = [None] * nb_inputs

        if not self.layers['i'].built:
            self.layers['i'].build(input_shape=(B, H))
        
        if not self.layers['e'].built:
            self.layers['e'].build(input_shape=(B, P, H))
        
        if not self.layers['v'].built:
            self.layers['v'].build(input_shape=(B, P, H))
        
        if not self.layers['g'].built:
            self.layers['g'].build(input_shape=(B, 2 * H))

    def step(self, inputs, states):
        vP_t = inputs
        hP_tm1 = states[0]
        vP = states[3]
        vP_mask = states[4]

        WP_v_Dot = self.layers['e'].call(vP)
        WPP_v_Dot = self.layers['i'].call(K.expand_dims(vP_t, axis=1))

        s_t_hat = K.tanh(WPP_v_Dot + WP_v_Dot)
        s_t = self.layers['v'].call(s_t_hat)
        s_t = K.batch_flatten(s_t)

        a_t = softmax(s_t, mask=vP_mask, axis=1)

        c_t = K.batch_dot(a_t, vP, axes=[1, 1])
        
        GRU_inputs = K.concatenate([vP_t, c_t])
        g = self.layers['g'].call(GRU_inputs)
        GRU_inputs = g * GRU_inputs
        
        hP_t, s = super(SelfAttnGRU, self).step(GRU_inputs, states)

        return hP_t, s
