from keras import backend as K

from keras import initializers
from keras import regularizers

from keras.engine.topology import Node
from keras.layers import Layer, InputLayer

class SharedWeightLayer(InputLayer):
    def __init__(self, 
                 size,
                 initializer='glorot_uniform',
                 regularizer=None,
                 **kwargs):
        self.size = tuple(size)
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)

        Layer.__init__(self, **kwargs)


        self.kernel = self.add_weight(shape=self.size,
                                      initializer=self.initializer,
                                      name='kernel',
                                      regularizer=self.regularizer)


        self.trainable = True
        self.built = True
        # self.sparse = sparse

        input_tensor = self.kernel * 1.0

        self.is_placeholder = False
        input_tensor._keras_shape = self.size
        
        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (self, 0, 0)

        Node(self,
            inbound_layers=[],
            node_indices=[],
            tensor_indices=[],
            input_tensors=[input_tensor],
            output_tensors=[input_tensor],
            input_masks=[None],
            output_masks=[None],
            input_shapes=[self.size],
            output_shapes=[self.size])
        
    def get_config(self):
        config = {
            'size': self.size,
            'initializer': initializers.serialize(self.initializer),
            'regularizer': regularizers.serialize(self.regularizer)
        }
        base_config = Layer.get_config(self)
        return dict(list(base_config.items()) + list(config.items()))

def SharedWeight(**kwargs):
    input_layer = SharedWeightLayer(**kwargs)

    outputs = input_layer.inbound_nodes[0].output_tensors
    if len(outputs) == 1: 
        return outputs[0]
    else:
        return outputs
