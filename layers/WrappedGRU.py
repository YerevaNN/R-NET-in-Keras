# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K
from keras.layers import Layer
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU

class WrappedGRU(GRU):

    def __init__(self, initial_state_provided=False, **kwargs):
        kwargs['implementation'] = kwargs.get('implementation', 2)
        assert(kwargs['implementation'] == 2)
        
        super(WrappedGRU, self).__init__(**kwargs)
        self.input_spec = None
        self.initial_state_provided = initial_state_provided


    def call(self, inputs, mask=None, training=None, initial_state=None):
        if self.initial_state_provided:
            initial_state = inputs[-1:]
            inputs = inputs[:-1]

            initial_state_mask = mask[-1:]
            mask = mask[:-1] if mask is not None else None

        self._non_sequences = inputs[1:]
        inputs = inputs[:1]

        self._mask_non_sequences = []
        if mask is not None:
            self._mask_non_sequences = mask[1:]
            mask = mask[:1]
        self._mask_non_sequences = [mask for mask in self._mask_non_sequences
                                    if mask is not None]

        if self.initial_state_provided:
            assert(len(inputs) == len(initial_state))
            inputs += initial_state

        if len(inputs) == 1:
            inputs = inputs[0]

        if isinstance(mask, list) and len(mask) == 1:
            mask = mask[0]

        return super(WrappedGRU, self).call(inputs, mask, training)

    def get_constants(self, inputs, training=None):
        constants = super(WrappedGRU, self).get_constants(inputs, training=training)
        constants += self._non_sequences
        constants += self._mask_non_sequences
        return constants

    def get_config(self):
        config = {'initial_state_provided': self.initial_state_provided}
        base_config = super(WrappedGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
