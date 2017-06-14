# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K
from keras.layers import Layer
from keras.layers.wrappers import TimeDistributed

from WrappedGRU import WrappedGRU

class PointerGRU(WrappedGRU):

    def build(self, input_shape):
        print('input_shape:', input_shape)
        H = self.units // 2
        assert(isinstance(input_shape, list))

        nb_inputs = len(input_shape)
        assert(nb_inputs == 3)

        assert(len(input_shape[0]) >= 2)
        B, T = input_shape[0][:2]

        assert(len(input_shape[1]) == 2)
        B, H_ = input_shape[1]
        assert(H_ == 2 * H)

        assert(len(input_shape[2]) == 3)
        B, P, H_ = input_shape[2]
        assert(H_ == 2 * H)

        self.input_spec = [None]
        super(PointerGRU, self).build(input_shape=(B, T, 2 * H))
        self.GRU_input_spec = self.input_spec
        self.input_spec = [None] * nb_inputs # TODO TODO TODO

        self.GRU_states = self.states
        self.states = [None, None]#, None]

        
        if not self.layers['s'].built:
            self.layers['s'].build(input_shape=(B, 2 * H))
        
        if not self.layers['e'].built:
            self.layers['e'].build(input_shape=(B, P, 2 * H))
        
        if not self.layers['v'].built:
            self.layers['v'].build(input_shape=(B, P, H))

    def step(self, inputs, states):
        # input
        ha_tm1 = states[0] # (B, 2H)
        hP = states[1] # (B, P, 2H)
        restStates = states[2:]

        WP_h_Dot = self.layers['e'].call(hP) # (B, P, H)
        Wa_h_Dot = self.layers['s'].call(K.expand_dims(ha_tm1, axis=1)) # (B, 1, H)

        s_t_hat = K.tanh(WP_h_Dot + Wa_h_Dot) # (B, P, H)
        s_t = self.layers['v'].call(s_t_hat) # (B, P, 1)
        s_t = K.batch_flatten(s_t) # (B, P)
        a_t = K.softmax(s_t) # (B, P)
        c_t = K.batch_dot(hP, a_t, axes=[1, 1]) # (B, 2H)

        GRU_inputs = c_t
        GRU_states = [ha_tm1] + list(restStates)
        ha_t, (ha_t_,) = super(PointerGRU, self).step(GRU_inputs, GRU_states)
        
        return a_t, [ha_t, hP]

    def compute_output_shape(self, input_shape):
        assert(isinstance(input_shape, list))

        nb_inputs = len(input_shape)
        assert(nb_inputs == 3)

        assert(len(input_shape[0]) >= 2)
        B, T = input_shape[0][:2]

        assert(len(input_shape[2]) == 3)
        B, P, H_ = input_shape[2]

        if self.return_sequences:
            return (B, T, P)
        else:
            return (B, P)