from keras import backend as K
from keras.engine.topology import Layer


class VariationalDropout(Layer):

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(VariationalDropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            symbolic_shape = K.shape(inputs)
            noise_shape = [shape if shape > 0 else symbolic_shape[axis]
                           for axis, shape in enumerate(self.noise_shape)]
            noise_shape = tuple(noise_shape)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape, seed=self.seed)

            return K.in_train_phase(dropped_inputs, inputs, training=training)

        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(VariationalDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
