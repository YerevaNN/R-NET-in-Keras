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

        assert(len(input_shape[1]) == 3)
        B, Q, H_ = input_shape[1]
        assert(H_ == 2 * H)

        self.input_spec = [None]
        super(QuestionAttnGRU, self).build(input_shape=(B, P, 4 * H))
        self.GRU_input_spec = self.input_spec
        self.input_spec = [None] * nb_inputs

    def step(self, inputs, states):
        uP_t = inputs
        vP_tm1 = states[0]
        _ = states[1:3] # ignore internal dropout/masks
        uQ, WQ_u, WP_v, WP_u, v, W_g1 = states[3:9]
        uQ_mask, = states[9:10]

        WQ_u_Dot = K.dot(uQ, WQ_u) #WQ_u
        WP_v_Dot = K.dot(K.expand_dims(vP_tm1, axis=1), WP_v) #WP_v
        WP_u_Dot = K.dot(K.expand_dims(uP_t, axis=1), WP_u) # WP_u

        s_t_hat = K.tanh(WQ_u_Dot + WP_v_Dot + WP_u_Dot)

        s_t = K.dot(s_t_hat, v) # v
        s_t = K.batch_flatten(s_t)
        a_t = softmax(s_t, mask=uQ_mask, axis=1)
        c_t = K.batch_dot(a_t, uQ, axes=[1, 1])

        GRU_inputs = K.concatenate([uP_t, c_t])
        g = K.dot(GRU_inputs, W_g1) # W_g1
        GRU_inputs = g * GRU_inputs
        vP_t, s = super(QuestionAttnGRU, self).step(GRU_inputs, states)

        return vP_t, s
