import numpy as np
from keras import backend as K
from keras.engine import Layer, InputSpec

class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.

    # Example

    ```python
        model = Sequential()
        model.add(Convolution2D(64, 3, 3,
                                border_mode='same',
                                input_shape=(3, 32, 32)))
        # now: model.output_shape == (None, 64, 32, 32)

        model.add(Flatten())
        # now: model.output_shape == (None, 65536)
    ```
    """

    def __init__(self, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)

    def compute_output_shape(self, input_shape):
        if all(input_shape[1:]):
            return (input_shape[0], np.prod(input_shape[1:]))
        return (input_shape[0], None)
        

    def call(self, inputs):
        return K.batch_flatten(inputs)