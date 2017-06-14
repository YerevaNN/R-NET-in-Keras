from keras import backend as K
from keras.metrics import categorical_accuracy
from keras.optimizers import Adadelta

from model import RNet
from data import BatchGen, load_dataset

from keras.layers import Masking

def categorical_accuracy_pair(y_true, y_pred):
    P = K.shape(y_true) [-1] // 2
    return categorical_accuracy(y_true[:, :P], y_pred[:, :P]) * categorical_accuracy(y_true[:, P:], y_pred[:, P:])

model = RNet()

model.compile(optimizer=Adadelta(lr=1.0),
              loss='categorical_crossentropy',
              metrics=['mse', categorical_accuracy_pair])

train_data = load_dataset('data/train_data.pkl')
valid_data = load_dataset('data/valid_data.pkl')

train_data_gen = BatchGen(*train_data, batch_size=2, shuffle=True)
valid_data_gen = BatchGen(*valid_data, batch_size=2, shuffle=False)

# model.predict_generator(generator=train_data_gen, 
#                         steps=train_data_gen.steps(),
#                         verbose=1)

print('Training...')

model.fit_generator(generator=train_data_gen,
                    steps_per_epoch=train_data_gen.steps(),
                    validation_data=valid_data_gen,
                    validation_steps=valid_data_gen.steps(),
                    epochs=100)


import cPickle as pickle
with open('dump.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)