import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from h5py import File as h5file

# For keras implementation
import keras
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np
import math

# Import shallow model
from models import EES_model, load_data_hdf5

def lr_schedule(epoch):
    if(epoch < 30):
        lr = 1e-2
    elif(epoch < 70):
        lr = 1e-3
    elif(epoch < 100):
        lr = 1e-4
    elif(epoch < 150):
        lr = 1e-5
    print("Learning rate: ", lr)
    return lr

""" train the model """
def train():
    # Hyperparameters
    batch_size = 128
    epochs = 150

    # Load data
    x_train, y_train = load_data_hdf5('train.h5')
    x_valid, y_valid = load_data_hdf5('valid.h5')

    print('training shape: ', x_train.shape, y_train.shape)
    print('testing shape: ', x_valid.shape, y_valid.shape)

    # Load the model
    inputs = Input(shape=(None,None,1))

    # create proposed model, with ResNet weights frozen
    model = EES_model(inputs=inputs)

    # dispaly model summary
    model.summary()

    rmsprop = RMSprop(lr=0.001)
    model.compile(optimizer=rmsprop, loss='mse', metrics=['mean_squared_error'])

    lr = LearningRateScheduler(lr_schedule)
    # Callbacks: (1) Checkpoint (2) Learning rate sheduler
    checkpoint = ModelCheckpoint(filepath="./tmp/weights.{epoch:03d}.h5", monitor='val_loss',save_best_only=True,
                                save_weights_only=True, mode='min')
    callbacks = [checkpoint,lr]

    # ------TRAINING----------
    model.fit(x_train,y_train, batch_size=batch_size, validation_data=(x_valid,y_valid),
                callbacks=callbacks, shuffle=True, epochs=epochs)

    score = model.evaluate(x_valid, y_valid, batch_size=batch_size)

    print("\nloss:",score[0])

""" train the model from saved checkpoint"""
def train_from_checkpoint():
    # Hyperparameters
    batch_size = 128
    epochs = 10

    # Load data
    x_train, y_train = load_data_hdf5('train.h5')
    x_valid, y_valid = load_data_hdf5('valid.h5')

    print('training shape: ', x_train.shape, y_train.shape)
    print('testing shape: ', x_valid.shape, y_valid.shape)

    # Load the model
    inputs = Input(shape=(None,None,1))

    # create proposed model, with ResNet weights frozen
    model = EES_model(inputs=inputs)
    model.load_weights('checkpoints/ees_weights.h5')

    # dispaly model summary
    model.summary()

    rmsprop = RMSprop(lr=1e-6)
    model.compile(optimizer=rmsprop, loss='mse', metrics=['mean_squared_error'])

    # Callbacks: (1) Checkpoint (2) Learning rate sheduler
    checkpoint = ModelCheckpoint(filepath="./tmp/weights.{epoch:03d}.h5", monitor='val_loss',save_best_only=True,
                                save_weights_only=True, mode='min')
    callbacks = [checkpoint]

    # ------TRAINING----------
    model.fit(x_train,y_train, batch_size=batch_size, validation_data=(x_valid,y_valid),
                callbacks=callbacks, shuffle=True, epochs=epochs)

    score = model.evaluate(x_valid, y_valid, batch_size=batch_size)

    model.save_weights('ees_weights.h5')

    print("\nloss:",score[0])

if __name__ == '__main__':
    #train()
    train_from_checkpoint()
