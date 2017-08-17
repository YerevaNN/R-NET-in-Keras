# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import argparse

import keras
from keras.callbacks import ModelCheckpoint

from model import RNet
from data import BatchGen, load_dataset

import sys
sys.setrecursionlimit(100000)

np.random.seed(10)

parser = argparse.ArgumentParser()
parser.add_argument('--hdim', default=75, help='Model to evaluate', type=int)
parser.add_argument('--batch_size', default=70, help='Batch size', type=int)
parser.add_argument('--nb_epochs', default=50, help='Number of Epochs', type=int)
parser.add_argument('--optimizer', default='Adadelta', help='Optimizer', type=str)
parser.add_argument('--lr', default=None, help='Learning rate', type=float)
parser.add_argument('--name', default='', help='Model dump name prefix', type=str)
parser.add_argument('--loss', default='categorical_crossentropy', help='Loss', type=str)

parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--char_level_embeddings', action='store_true')

parser.add_argument('--train_data', default='data/train_data.pkl', help='Train Set', type=str)
parser.add_argument('--valid_data', default='data/valid_data.pkl', help='Validation Set', type=str)

# parser.add_argument('model', help='Model to evaluate', type=str)
args = parser.parse_args()

print('Creating the model...', end='')
model = RNet(hdim=args.hdim, dropout_rate=args.dropout, N=300, M=30,
             char_level_embeddings=args.char_level_embeddings)
print('Done!')

print('Compiling Keras model...', end='')
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr} if args.lr else {}}
model.compile(optimizer=optimizer_config,
              loss=args.loss,
              metrics=['accuracy'])
print('Done!')

print('Loading datasets...', end='')
train_data = load_dataset(args.train_data)
valid_data = load_dataset(args.valid_data)
print('Done!')

print('Preparing generators...', end='')
maxlen = [300, 300, 30, 30] if args.char_level_embeddings else [300, 30]

train_data_gen = BatchGen(*train_data, batch_size=args.batch_size, shuffle=False, group=True, maxlen=maxlen)
valid_data_gen = BatchGen(*valid_data, batch_size=args.batch_size, shuffle=False, group=True, maxlen=maxlen)
print('Done!')

print('Training...', end='')

path = 'models/' + args.name + '{epoch}-t{loss}-v{val_loss}.model'

model.fit_generator(generator=train_data_gen,
                    steps_per_epoch=train_data_gen.steps(),
                    validation_data=valid_data_gen,
                    validation_steps=valid_data_gen.steps(),
                    epochs=args.nb_epochs,
                    callbacks=[
                        ModelCheckpoint(path, verbose=1, save_best_only=True)
                    ])
print('Done!')
