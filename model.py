# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, RepeatVector
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.pooling import GlobalMaxPooling1D

from layers import QuestionAttnGRU
from layers import SelfAttnGRU
from layers import PointerGRU
from layers import QuestionPooling
from layers import Flatten

class RNet(Model):
    def __init__(self, **kwargs):
        '''Dimensions'''
        self.B = None
        self.N = None
        self.M = None
        self.H = 75
        self.W = 300

        self.shared_layers = {
            'v':        Dense(units=1, use_bias=False, name='v'),
            'WQ_u':     Dense(units=self.H, use_bias=False, name='WQ_u'),
            'WP_u':     Dense(units=self.H, use_bias=False, name='WP_u'),
            'WP_v':     Dense(units=self.H, use_bias=False, name='WP_v'),
            'W_g1':     Dense(units=4 * self.H, use_bias=False, name='W_g1'),
            'W_g2':     Dense(units=2 * self.H, use_bias=False, name='W_g2'),
            'WP_h':     Dense(units=self.H, use_bias=False, name='WP_h'),
            'Wa_h':     Dense(units=self.H, use_bias=False, name='Wa_h'),
            'WPP_v':    Dense(units=self.H, use_bias=False, name='WPP_v'),
            'WQ_v':     Dense(units=self.H, use_bias=False, name='WQ_v'),
        }

        self.P = Input(shape=(self.N, self.W), name='P')
        self.Q = Input(shape=(self.M, self.W), name='Q')

        self.uP = self.P
        for i in range(3):
            self.uP = Bidirectional(GRU(units=self.H,
                                        return_sequences=True)) (self.uP)

        self.uQ = self.Q
        for i in range(3):
            self.uQ = Bidirectional(GRU(units=self.H,
                                        return_sequences=True)) (self.uQ)

        self.vP = QuestionAttnGRU(units=self.H,
                                  return_sequences=True,
                                  layer_map={
                                      'e': 'WQ_u',
                                      's': 'WP_v',
                                      'i': 'WP_u',
                                      'v': 'v',
                                      'g': 'W_g1'
                                  },
                                  layers=self.shared_layers,
                                  name='vP') ([
                                      self.uP, self.uQ
                                  ])

        self.hP_forward = SelfAttnGRU(units=self.H,
                                      return_sequences=True,
                                      layer_map={
                                          'e': 'WP_v',
                                          'i': 'WPP_v',
                                          'v': 'v',
                                          'g': 'W_g2'
                                      },
                                      layers=self.shared_layers) ([
                                          self.vP, self.vP
                                      ])

        self.hP_backward = SelfAttnGRU(units=self.H,
                                       return_sequences=True,
                                       layer_map={
                                           'e': 'WP_v',
                                           'i': 'WPP_v',
                                           'v': 'v',
                                           'g': 'W_g2'
                                       },
                                       layers=self.shared_layers,
                                       go_backwards=True) ([
                                           self.vP, self.vP
                                       ])

        self.hP = Concatenate(name='hP') ([
            self.hP_forward, self.hP_backward
        ])

        self.rQ = QuestionPooling(layer_map={
                                      'e': 'WQ_u',
                                      's': 'WQ_v',
                                      'v': 'v'
                                  },
                                  layers=self.shared_layers,
                                  name='rQ') (self.uQ)

        self.fake_input = GlobalMaxPooling1D() (self.P)
        self.fake_input = RepeatVector(n=2,
                                       name='fake_input') (self.fake_input)

        self.ps = PointerGRU(units=2 * self.H,
                             return_sequences=True,
                             layer_map={
                                 'e': 'WP_h',
                                 's': 'Wa_h',
                                 'v': 'v'
                             },
                             layers=self.shared_layers,
                             name='ps') ([
                                 self.fake_input, self.rQ, self.hP
                             ])
                             
        print(K.int_shape(self.ps))
        self.ps = Flatten(name='ps_flat') (self.ps)

        self.inputs = [self.P, self.Q]

        self.outputs = [self.ps]

        super(RNet, self).__init__(inputs=self.inputs,
                                   outputs=self.outputs,
                                   **kwargs)

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        weights = super(RNet, self).trainable_weights
        for layer in self.shared_layers.values():
            weights += layer.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = super(RNet, self).non_trainable_weights
        for layer in self.shared_layers.values():
            weights += layer.non_trainable_weights
        if not self.trainable:
            trainable_weights = super(RNet, self).trainable_weights
            for layer in self.shared_layers.values():
                trainable_weights += layer.trainable_weights
            return trainable_weights + weights
        return weights

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, weights):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config, custom_objects=None):
        raise NotImplementedError

    def save(self, filepath, overwrite=True, include_optimizer=True):
        raise NotImplementedError

    def save_weights(self, filepath, overwrite=True):
        raise NotImplementedError

    def load_weights(self, filepath, by_name=False):
        raise NotImplementedError

    def to_json(self, **kwargs):
        raise NotImplementedError

    def to_yaml(self, **kwargs):
        raise NotImplementedError

    def summary(self, line_length=None, positions=None):
        raise NotImplementedError
