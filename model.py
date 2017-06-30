# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, RepeatVector, Masking, Dropout
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.pooling import GlobalMaxPooling1D

from layers import QuestionAttnGRU
from layers import SelfAttnGRU
from layers import PointerGRU
from layers import QuestionPooling
from layers import Flatten
from layers import SharedWeight

class RNet(Model):
    def __init__(self, **kwargs):
        '''Dimensions'''
        B = None
        N = None
        M = None
        H = 75
        W = 300

        v = SharedWeight(size=(H, 1), name='v')
        WQ_u = SharedWeight(size=(2 * H, H), name='WQ_u')
        WP_u = SharedWeight(size=(2 * H, H), name='WP_u')
        WP_v = SharedWeight(size=(H, H), name='WP_v')
        W_g1 = SharedWeight(size=(4 * H, 4 * H), name='W_g1')
        W_g2 = SharedWeight(size=(2 * H, 2 * H), name='W_g2')
        WP_h = SharedWeight(size=(2 * H, H), name='WP_h')
        Wa_h = SharedWeight(size=(2 * H, H), name='Wa_h')
        WQ_v = SharedWeight(size=(2 * H, H), name='WQ_v')
        WPP_v = SharedWeight(size=(H, H), name='WPP_v')

        shared_weights = [v, WQ_u, WP_u, WP_v, W_g1, W_g2, WP_h, Wa_h, WQ_v, WPP_v]
            
        P = Input(shape=(N, W), name='P')
        Q = Input(shape=(M, W), name='Q')
        
        uP = Masking() (P)
        for i in range(3):
            uP = Bidirectional(GRU(units=H,
                                   return_sequences=True,
                                   dropout=0.2)) (uP)
        uP = Dropout(rate=0.2, name='uP') (uP)
        
        uQ = Masking() (Q)
        for i in range(3):
            uQ = Bidirectional(GRU(units=H,
                                   return_sequences=True, 
                                   dropout=0.2)) (uQ)
        uQ = Dropout(rate=0.2, name='uQ') (uQ)

        vP = QuestionAttnGRU(units=H,
                             return_sequences=True) ([
                                 uP, uQ, 
                                 WQ_u, WP_v, WP_u, v, W_g1
                             ])
        vP = Dropout(rate=0.2, name='vP') (vP)

        hP = Bidirectional(SelfAttnGRU(units=H,
                                       return_sequences=True)) ([
                                          vP, vP,
                                          WP_v, WPP_v, v, W_g2
                                      ])

        hP = Dropout(rate=0.2, name='hP') (hP)

        rQ = QuestionPooling() ([uQ, WQ_u, WQ_v, v])
        rQ = Dropout(rate=0.2, name='rQ') (rQ)

        fake_input = GlobalMaxPooling1D() (P)
        fake_input = RepeatVector(n=2, name='fake_input') (fake_input)

        ps = PointerGRU(units=2 * H,
                        return_sequences=True,
                        initial_state_provided=True,
                        name='ps') ([
                            fake_input, hP,
                            WP_h, Wa_h, v, 
                            rQ
                        ])
        
        ps = Flatten(name='ps_flat') (ps)

        inputs = [P, Q] + shared_weights

        outputs = [ps]

        super(RNet, self).__init__(inputs=inputs,
                                   outputs=outputs,
                                   **kwargs)
