import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from h5py import File as h5file

# For keras implementation
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D,Input,BatchNormalization,Activation,Conv2DTranspose,MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import numpy as np


"""
[1] End-to-End Image Super-Resolution via Deep and Shallow Convolutional Networks
[2] Enhanced Deep Residual Networks for Single Image Super-Resolution
"""

""" Deep layers [1] """
def EED_model(inputs):
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
    y = Conv2DTranspose(filters=4, kernel_size=(16, 16), strides=(4, 4), padding='same', activation='relu')(y)
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
    output = Activation('linear')(output)   # regression

    model = Model(inputs=inputs,outputs=output)

    model.name = 'EED_model'

    return model

""" Shallow layers [1] """
def EES_model(inputs):
    y = conv_bn_relu(inputs, 3, (3,3))
    y = Conv2DTranspose(filters=4, kernel_size=(16, 16), strides=(4, 4), padding='same', activation='relu')(y)
    y = conv_bn_relu(y, 5, (1,1))

    output = Conv2D(1, (1,1), kernel_initializer='glorot_uniform', padding='same', use_bias=True)(y)
    output = Activation('linear')(output)   # regression

    model = Model(inputs=inputs,outputs=output)

    model.name = 'EES_model'

    return model

""" Deep and Shallow Network [1] """
def EEDS_model(inputs, freeze_weights=False):
    ees_model = EES_model(inputs=inputs)
    eed_model = EED_model(inputs=inputs)

    if(freeze_weights):
        ees_weights = model.load_weights('ees_weights.h5')
        eed_weights = model.load_weights('eed_weights.h5')
        ees_model.set_weights(ees_weights)
        eed_model.set_weights(eed_weights)
        for ees_layer,eed_layer in zip(ees_model,eed_model):
            ees_layer.trainable = False
            eed_layer.trainable = False

    # Remove the linear activation layers
    ees_model.layers.pop()
    eed_model.layers.pop()

    add_layer = keras.layers.add([ees_model.layers[-1].output, eed_model.layers[-1].output])

    y = res_block(add_layer, 16,depth=3)

    output = Conv2D(1, (1,1), kernel_initializer='glorot_uniform', padding='same', use_bias=True)(y)
    output = Activation('linear')(output)   # regression

    model = Model(inputs=inputs,outputs=output)

    model.name = 'EEDS_model'

    return model

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

""" combined conv, batch norm and activation = 'relu' """
def conv_bn_relu(inputs, filters, kernel_size, padding='same'):
    y = Conv2D(filters,kernel_size,padding=padding,
               kernel_initializer='glorot_uniform',
               use_bias=True)(inputs)
    y = bn_relu(y)
    return y

""" Residual block implemented in [2] """
def res_block(inputs, filters, padding='same', depth=1):

    # Alternate 1x1-3x3
    for i in range(depth):
        if(i == 0):
            previous = inputs
        y = Conv2D(filters,(1,1),padding=padding,
                   kernel_initializer='glorot_uniform',
                   use_bias=True)(previous)
        y = Activation('relu')(y)
        y = Conv2D(filters,(3,3),padding=padding,
                   kernel_initializer='glorot_uniform',
                   use_bias=True)(y)

        previous = keras.layers.add([previous,y])

    output = previous

    return output

def show_models():
    # Load data
    x_train, y_train = load_data_hdf5('train.h5')
    x_valid, y_valid = load_data_hdf5('valid.h5')

    # Load the model
    inputs = Input(shape=(24,24,1))

    model1 = EES_model(inputs=inputs)
    model2 = EED_model(inputs=inputs)
    model3 = EEDS_model(inputs=inputs)

    input('Press Enter to show Shallow Model')
    model1.summary()
    input('Press Enter to show Deep Model')
    model2.summary()
    input('Press Enter to show Combination Model')
    model3.summary()

if __name__ == '__main__':
    show_models()
