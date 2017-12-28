import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from h5py import File as h5file

# For keras implementation
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Input, BatchNormalization, Activation, Conv2DTranspose, MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import numpy as np
import math

# Imported models
from keras.applications.xception import Xception

""" load an h5py file dataset """
def load_data_hdf5(filename):
    with h5file(filename,'r') as dataset:
        data1 = np.array(dataset.get('data'))
        data2 = np.array(dataset.get('label'))
        # for scale = 4: 24x24
        data1 = np.reshape(data1, (-1,24,24,1))
        data2 = np.reshape(data2, (-1,96,96,1))
        return data1, data2

""" combined batch norm and activation ='relu' """
def bn_relu(layer):
    y = BatchNormalization()(layer)
    y = Activation('relu')(y)
    return y

def conv_bn_relu(inputs, filters, kernel_size, padding='same'):
    y = Conv2D(filters,kernel_size,padding=padding,
               kernel_initializer='glorot_uniform',
               use_bias=True)(inputs)
    y = bn_relu(y)

    return y

""" Proposed model """
def SR_model(inputs):
    """ Feature Extraction """
    # Replaced with inception module
    tower_0 = conv_bn_relu(inputs, 64, (1,1))

    tower_1 = conv_bn_relu(inputs,64,(1,1))
    tower_1 = conv_bn_relu(tower_1, 64, (3,3))

    tower_2 = conv_bn_relu(inputs,64,(1,1))
    tower_2 = conv_bn_relu(tower_2, 64, (5,5))

    tower_3 = MaxPooling2D((3,3),strides=(1,1), padding='same')(inputs)
    tower_3 = conv_bn_relu(tower_3, 64, (1,1))

    feature = keras.layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=-1)

    """ Upsampling """
    y = conv_bn_relu(feature, 4, (1,1))
    for _ in range(6):
        y = Conv2DTranspose(4, (13,13), kernel_initializer='glorot_uniform', padding='valid', use_bias=True)(y)
        y = Activation('relu')(y)
    y = conv_bn_relu(y, 64,(1,1))

    """ Reconstruction """
    # ResNet v2 implementation
    layer1 = conv_bn_relu(y, 64, (1,1))
    layer2 = conv_bn_relu(layer1, 64, (3,3))
    layer3 = Conv2D(64, (1,1), kernel_initializer='glorot_uniform', padding='same', use_bias=True)(layer2)
    add1 = keras.layers.add([layer1,layer3])

    layer1 = conv_bn_relu(add1, 64, (1,1))
    layer2 = conv_bn_relu(layer1, 64, (3,3))
    layer3 = Conv2D(64, (1,1), kernel_initializer='glorot_uniform', padding='same', use_bias=True)(layer2)
    add2 = keras.layers.add([layer1,layer3])

    # Multiscale
    init_multi = conv_bn_relu(add2, 16, (1,1))
    scale_1 = conv_bn_relu(init_multi, 16, (1,1))
    scale_3 = conv_bn_relu(init_multi, 16, (3,3))
    scale_5 = conv_bn_relu(init_multi, 16, (5,5))
    scale_7 = conv_bn_relu(init_multi, 16, (7,7))
    scale_layer = keras.layers.concatenate([scale_1, scale_3, scale_5, scale_7])

    output = Conv2D(1, (1,1), kernel_initializer='glorot_uniform', padding='same', use_bias=True)(scale_layer)

    model = Model(inputs=inputs,outputs=output)

    return model

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
    batch_size = 16
    epochs = 10

    # Load data
    x_train, y_train = load_data_hdf5('train.h5')
    x_valid, y_valid = load_data_hdf5('valid.h5')

    print('training shape: ',x_train.shape, y_train.shape)
    print('testing shape: ',x_valid.shape, y_valid.shape)

    # Load the model
    inputs = Input(shape=(24,24,1))

    # create proposed model, with ResNet weights frozen
    model = SR_model(inputs=inputs)

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

    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    print("\nloss:",score[0])

if __name__ == '__main__':
    train()
