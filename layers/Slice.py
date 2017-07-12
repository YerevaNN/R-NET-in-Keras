import numpy as np
from keras import backend as K
from keras.engine import Layer, InputSpec

class Slice(Layer):
    def __init__(self, indices, axis=1, **kwargs):
        self.supports_masking = True
        self.axis = axis
        
        if isinstance(indices, slice):
            self.indices = (indices.start, indices.stop, indices.step)
        else:
            self.indices = indices

        self.slices = [ slice(None) ] * self.axis

        if isinstance(self.indices, int):
            self.slices.append(self.indices)
        elif isinstance(self.indices, (list, tuple)):
            self.slices.append(slice(*self.indices))
        else:
            raise TypeError("indices must be int or slice")
        
        super(Slice, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        return inputs[self.slices]

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        for i, slice in enumerate(self.slices):
            if i == self.axis:
                continue
            start = slice.start or 0
            stop = slice.stop or input_shape[i]
            step = slice.step or 1
            input_shape[i] = None if stop is None else (stop - start) // step
        del input_shape[self.axis]

        return tuple(input_shape)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return mask
        if self.axis == 1:
            return mask[self.slices]
        else:
            return mask

    def get_config(self):
        config = {'axis': self.axis,
                  'indices': self.indices}
        base_config = super(Slice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
