from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import cPickle as pickle

from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing import sequence

from random import shuffle
import itertools

def load_dataset(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def padded_batch_input(input, indices=None, dtype=K.floatx(), maxlen=None):
    if indices is None:
        indices = np.arange(len(input))

    batch_input = [input[i] for i in indices]
    return sequence.pad_sequences(batch_input, maxlen, dtype, padding='post')

def categorical_batch_target(target, classes, indices=None, dtype=K.floatx()):
    if indices is None:
        indices = np.arange(len(target))

    batch_target = [min(target[i], classes-1) for i in indices]
    return np_utils.to_categorical(batch_target, classes).astype(dtype)

def lengthGroup(length):
    if length < 150:
        return 0
    if length < 240:
        return 1
    if length < 380:
        return 2
    if length < 520:
        return 3
    if length < 660:
        return 4
    return 5

class BatchGen(object):
    def __init__(self, inputs, targets=None, batch_size=None, stop=False,
                 shuffle=True, balance=False, dtype=K.floatx(),
                 flatten_targets=False, sort_by_length=False,
                 group=False, maxlen=None):
        assert len(set([len(i) for i in inputs])) == 1
        assert(not shuffle or not sort_by_length)
        self.inputs = inputs
        self.nb_samples = len(inputs[0])

        self.batch_size = batch_size if batch_size else self.nb_samples

        self.dtype = dtype

        self.stop = stop
        self.shuffle = shuffle
        self.balance = balance
        self.targets = targets
        self.flatten_targets = flatten_targets
        if isinstance(maxlen, (list, tuple)):
            self.maxlen = maxlen
        else:
            self.maxlen = [maxlen] * len(inputs)

        self.sort_by_length = None
        if sort_by_length:
            self.sort_by_length = np.argsort([-len(p) for p in inputs[0]])

        # if self.targets and self.balance:
        #     self.class_weight = class_weight(self.targets)

        self.generator = self._generator()
        self._steps = -(-self.nb_samples // self.batch_size) # round up

        self.groups = None
        if group is not False:
            indices = np.arange(self.nb_samples)

            ff = lambda i: lengthGroup(len(inputs[0][i]))

            indices = np.argsort([ff(i) for i in indices])

            self.groups = itertools.groupby(indices, ff)

            self.groups = {k: np.array(list(v)) for k, v in self.groups}

    def _generator(self):
        while True:
            if self.shuffle:
                permutation = np.random.permutation(self.nb_samples)
            elif self.sort_by_length is not None:
                permutation = self.sort_by_length
            elif self.groups is not None:
                # permutation = np.arange(self.nb_samples)
                # tmp = permutation.copy()
                # for id in self.group_ids:
                #     mask = (self.groups==id)
                #     tmp[mask] = np.random.permutation(permutation[mask])
                # permutation = tmp
                # import ipdb
                # ipdb.set_trace()

                for k, v in self.groups.items():
                    np.random.shuffle(v)

                tmp = np.concatenate(self.groups.values())
                batches = np.array_split(tmp, self._steps)

                remainder = []
                if len(batches[-1]) < self._steps:
                    remainder = batches[-1:]
                    batches = batches[:-1]

                shuffle(batches)
                batches += remainder
                permutation = np.concatenate(batches)

            else:
                permutation = np.arange(self.nb_samples)

            i = 0
            longest = 767

            while i < self.nb_samples:
                if self.sort_by_length is not None:
                    bs = self.batch_size * 767 // self.inputs[0][permutation[i]].shape[0]
                else:
                    bs = self.batch_size
                
                indices = permutation[i : i + bs]
                i = i + bs

            # for i in range(0, self.nb_samples, self.batch_size):
                # indices = permutation[i : i + self.batch_size]

                batch_X = [padded_batch_input(x, indices, self.dtype, maxlen)
                           for x, maxlen in zip(self.inputs, self.maxlen)]

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

                # batch_W = np.array([self.class_weight[y] for y in batch_targets])
                batch_W = np.array([bs / self.batch_size for x in batch_X[0]]).astype(self.dtype)
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
        if self.sort_by_length is None: 
            return self._steps

        print("Steps was called")
        if self.shuffle:
            permutation = np.random.permutation(self.nb_samples)
        elif self.sort_by_length is not None:
            permutation = self.sort_by_length
        else:
            permutation = np.arange(self.nb_samples)

        i = 0
        longest = 767

        self._steps = 0
        while i < self.nb_samples:
            bs = self.batch_size * 767 // self.inputs[0][permutation[i]].shape[0]
            i = i + bs
            self._steps += 1

        return self._steps

batch_gen = BatchGen # for backward compatibility
