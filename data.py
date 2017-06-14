from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import cPickle as pickle

from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing import sequence

def load_dataset(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def class_weight(dataset):
    labels = dataset[1] if isinstance(dataset, tuple) else dataset
    labels = list(labels)
    nb_samples = len(labels)
    return {
        0: nb_samples / labels.count(0),
        1: nb_samples / labels.count(1)
    }

def padded_batch_input(input, indices=None, dtype=K.floatx()):
    if indices is None:
        indices = np.arange(len(input))

    batch_input = [input[i] for i in indices]
    return sequence.pad_sequences(batch_input, dtype=dtype, padding='post')

def categorical_batch_target(target, classes, indices=None, dtype=K.floatx()):
    if indices is None:
        indices = np.arange(len(target))

    batch_target = [min(target[i], classes-1) for i in indices]
    return np_utils.to_categorical(batch_target, classes).astype(dtype)

class BatchGen(object):
    def __init__(self, inputs, targets=None, batch_size=None, stop=False,
                 shuffle=True, balance=False, dtype=K.floatx(),
                 flatten_targets=True):
        assert len(set([len(i) for i in inputs])) == 1
        self.inputs = inputs
        self.nb_samples = len(inputs[0])

        self.batch_size = batch_size if batch_size else self.nb_samples

        self.dtype = dtype

        self.stop = stop
        self.shuffle = shuffle
        self.balance = balance
        self.targets = targets
        self.flatten_targets = flatten_targets
        if self.targets and self.balance:
            self.class_weight = class_weight(self.targets)

        self.generator = self._generator()
        self._steps = -(-self.nb_samples // self.batch_size) # round up

    def _generator(self):
        while True:
            if self.shuffle:
                permutation = np.random.permutation(self.nb_samples)
            else:
                permutation = np.arange(self.nb_samples)

            for i in range(0, self.nb_samples, self.batch_size):
                indices = permutation[i : i + self.batch_size]

                batch_X = [padded_batch_input(input, indices, self.dtype)
                           for input in self.inputs]

                P = batch_X[0].shape[1]

                if not self.targets:
                    yield batch_X
                    continue

                batch_Y = [categorical_batch_target(target, P,
                                                    indices, self.dtype)
                           for target in self.targets]

                if self.flatten_targets:
                    batch_Y = np.concatenate(batch_Y, axis=-1)

                if not self.balance:
                    yield (batch_X, batch_Y)
                    continue

                batch_W = np.array([self.class_weight[y] for y in batch_targets])
                yield (batch_X, batch_Y, batch_W)

            if self.stop:
                raise StopIteration

    def __iter__(self):
        return self.generator

    def next(self):
        return self.generator.next()

    def __next__(self):
        return self.generator.__next__()

    def steps(self):
        return self._steps

batch_gen = BatchGen # for backward compatibility
