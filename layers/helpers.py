# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras import backend as K

def softmax(x, axis, mask=None):
    if mask is None:
        mask = K.constant(True)
    mask = K.cast(mask, K.floatx())
    if K.ndim(x) is K.ndim(mask) + 1:
        mask = K.expand_dims(mask)

    m = K.max(x, axis=axis, keepdims=True)
    e = K.exp(x - m) * mask
    s = K.sum(e, axis=axis, keepdims=True)
    s += K.cast(K.cast(s < K.epsilon(), K.floatx()) * K.epsilon(), K.floatx())
    return e / s

def compute_mask(x, mask_value=0):
    boolean_mask = K.any(K.not_equal(x, mask_value), axis=-1, keepdims=False)
    return K.cast(boolean_mask, K.floatx())