# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K
from keras.layers import Layer
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU

class WrappedGRU(GRU):

    def __init__(self, layer_map, layers=None, **kwargs):
        kwargs['implementation'] = kwargs.get('implementation', 2)
        assert(kwargs['implementation'] == 2)
        
        super(WrappedGRU, self).__init__(**kwargs)
        self.input_spec = None
        self.layer_map = layer_map
        if layers is not None:
            self.set_layers(layers)


    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._non_sequences = inputs[-1:]
        self._mask_non_sequences = mask[-1:] if isinstance(mask, list) else []

        inputs = inputs[:-1]
        if(len(inputs) == 1):
            inputs = inputs[0]
        
        if mask is not None:
            mask = mask[:-1]
            if(len(mask) == 1):
                mask = mask[0]
        return super(WrappedGRU, self).call(inputs, mask, training)

    def get_constants(self, inputs, training=None):
        constants = super(WrappedGRU, self).get_constants(inputs, training=training)
        constants += self._non_sequences
        constants += self._mask_non_sequences
        return constants

    @property
    def layers(self):
        return {k: self._layers[v] for k, v in self.layer_map.iteritems()}

    def set_layers(self, layers):
        self._layers = layers

    def get_config(self):
        # layers_config = { name: {
        #                     'class_name': layer.__class__.__name__,
        #                     'config': layer.get_config()
        #                   } for name, layer in self.layers.iteritems() }
        config = {'layer_map': self.layer_map}
        base_config = super(WrappedGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # @classmethod
    # def from_config(cls, config, custom_objects=None):
    #     from keras.layers import deserialize
    #     # layers_config = config.pop('layers')
    #     # layers = {
    #     #     name: deserialize(layer, custom_objects=custom_objects)
    #     #     for name, layer_config in layers_config.iteritems()
    #     # }
    #     return cls(**config)
    
    # @property
    # def trainable_weights(self):
    #     trainable_weights = super(WrappedGRU, self).trainable_weights
    #     for name, layer in sorted(self.layers.items()):
    #         trainable_weights += layer.trainable_weights
    #     return trainable_weights

    # @property
    # def non_trainable_weights(self):
    #     non_trainable_weights = super(WrappedGRU, self).non_trainable_weights
    #     for name, layer in sorted(self.layers.items()):
    #         non_trainable_weights += layer.non_trainable_weights
    #     return non_trainable_weights

    # def get_weights(self):
    #     weights = super(WrappedGRU, self).get_weights()
    #     for name, layer in sorted(self.layers.items()):
    #         weights += layer.get_weights()
    #     return weights

    # def set_weights(self, weights):
    #     values = weights[:len(self.weights)]
    #     del weights[:len(self.weights)]
    #     super(WrappedGRU, self).set_weights(values)

    #     for name, layer in sorted(self.layers.items()):
    #         values = weights[:len(layer.weights)]
    #         del weights[:len(layer.weights)]
    #         layer.set_weights(values)
