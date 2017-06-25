# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K

from WrappedGRU import WrappedGRU
from helpers import compute_mask, softmax

class QuestionAttnGRU(WrappedGRU):

    def build(self, input_shape):
        H = self.units
        assert(isinstance(input_shape, list))
        
        nb_inputs = len(input_shape)
        assert(nb_inputs >= 2)

        assert(len(input_shape[0]) == 3)
        B, P, H_ = input_shape[0]
        assert(H_ == 2 * H)

        # assert(len(input_shape[1]) == 2)
        # B, H_ = input_shape[1]
        # assert(H_ == H)

        assert(len(input_shape[-1]) == 3)
        B, Q, H_ = input_shape[-1]
        assert(H_ == 2 * H)

        self.input_spec = [None]
        super(QuestionAttnGRU, self).build(input_shape=(B, P, 4 * H))
        self.GRU_input_spec = self.input_spec
        self.input_spec = [None] * nb_inputs

        if not self.layers['i'].built:
            self.layers['i'].build(input_shape=(B, 2 * H))
        
        if not self.layers['s'].built:
            self.layers['s'].build(input_shape=(B, H))
        
        if not self.layers['e'].built:
            self.layers['e'].build(input_shape=(B, Q, 2 * H))
        
        if not self.layers['v'].built:
            self.layers['v'].build(input_shape=(B, Q, H))
        
        if not self.layers['g'].built:
            self.layers['g'].build(input_shape=(B, 4 * H))

    def step(self, inputs, states):
        uP_t = inputs
        vP_tm1 = states[0]
        uQ = states[3]
        uQ_mask = states[4]

        WQ_u_Dot = self.layers['e'].call(uQ)
        WP_v_Dot = self.layers['s'].call(K.expand_dims(vP_tm1, axis=1))
        WP_u_Dot = self.layers['i'].call(K.expand_dims(uP_t, axis=1))

        s_t_hat = K.tanh(WQ_u_Dot + WP_v_Dot + WP_u_Dot)
        # s_t_hat *= 

        s_t = self.layers['v'].call(s_t_hat)
        s_t = K.batch_flatten(s_t)
        a_t = softmax(s_t, mask=uQ_mask, axis=1)
        c_t = K.batch_dot(a_t, uQ, axes=[1, 1])

        GRU_inputs = K.concatenate([uP_t, c_t])
        g = self.layers['g'].call(GRU_inputs)
        GRU_inputs = g * GRU_inputs
        vP_t, s = super(QuestionAttnGRU, self).step(GRU_inputs, states)

        return vP_t, s
