import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from h5py import File as h5file

# For keras implementation
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D,Input,BatchNormalization,Activation,Conv2DTranspose
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import numpy as np
import math

# Imported models
from keras.applications.resnet50 import ResNet50
from inception_resnet_v2 import InceptionResNetV2

""" load an h5py file dataset """
def load_data_hdf5(filename):
    with h5file(filename,'r') as dataset:
        data = np.array(dataset.get('data'))
        label = np.array(dataset.get('label'))
        data = np.reshape(data,(-1,32,32,1))
        label = np.reshape(label,(-1,20,20,1))
        return data, label

""" PSNR metric for images """
def psnr_measure(target, ref):
    # assume RGB image
    diff = np.array(ref, dtype=float) - np.array(target, dtype=float)
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2))

    return 20 * math.log10(255. / rmse)

""" combined batch norm and activation ='relu' """
def bn_activation(layer):
    y = BatchNormalization()(layer)
    y = Activation('relu')(y)
    return y

""" Proposed model """
def SR_model(inputs, freeze_resnet=False):
    """ DECONV LAYERS """
    # First 15 layers: 1 filter, (9x9)
    y = Conv2DTranspose(filters=1, kernel_size=(9,9), kernel_initializer='glorot_uniform',
                        padding='valid', use_bias=True)(inputs)
    y = bn_activation(y)

    for _ in range(14):
        y = Conv2DTranspose(filters=1, kernel_size=(9,9), kernel_initializer='glorot_uniform',
                        padding='valid', use_bias=True)(y)
        y = bn_activation(y)

    # Next 18 layers: Decreasing number of filters , (6x6)
    for i in range(18):
        if(i < 6):
            num_filters = 64
        elif(i < 12):
            num_filters = 32
        elif(i < 18):
            num_filters = 16
        y = Conv2DTranspose(filters=num_filters, kernel_size=(6,6), kernel_initializer='glorot_uniform',
                            padding='valid', use_bias=True)(y)
        y = bn_activation(y)

    # Last 5 Layers: alternate (3x3) and (1x1) layers
    for i in range(15):
        if(i % 2 == 0): #even
            kernel_size = (1,1)
        else: #odd
            kernel_size = (3,3)
        y = Conv2DTranspose(filters=4, kernel_size=kernel_size, kernel_initializer='glorot_uniform',
                            padding='valid', use_bias=True)(y)
        y = bn_activation(y)

    # Last Layers: To connect to ResNet50
    y = Conv2DTranspose(filters=3, kernel_size=(1,1), kernel_initializer='glorot_uniform',
                        padding='valid', use_bias=True)(y)
    upscale_output = bn_activation(y)

    upscale_model = Model(inputs=inputs,outputs=upscale_output)

    """ RESNET """
    ResNet50_model = ResNet50(include_top=False,
                              weights='imagenet',
                              input_tensor=Input(shape=(256,256,3)),
                              pooling='max')

    # Remove pool and classification layers
    for _ in range(2):
        ResNet50_model.layers.pop()

    # Freeze weights of resnet50
    if(freeze_resnet):
        for layer in ResNet50_model.layers:
            layer.trainable = False

    """ DECONV AGAIN """
    y = Conv2DTranspose(filters=3, kernel_size=(9,9), kernel_initializer='glorot_uniform',
                        padding='valid', use_bias=True)(ResNet50_model.layers[-1].output)
    y = Conv2DTranspose(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform',
                        padding='valid', use_bias=True)(y)

    resnet_model = Model(inputs=ResNet50_model.input, outputs=y)

    # Connect the two models
    model = Model(inputs=inputs,outputs=resnet_model(upscale_model.output))

    return model

""" train the model """
def train():
    # Hyperparameters
    batch_size = 1
    epochs = 10

    # Load data
    x_train, y_train = load_data_hdf5('train.h5')
    x_test, y_test = load_data_hdf5('test.h5')

    print('training shape: ',x_train.shape, y_train.shape)
    print('testing shape: ',x_test.shape, y_test.shape)

    # Load the model
    inputs = Input(shape=(32,32,1))

    # Append ResNet50 in Upscale model
    model = SR_model(inputs=inputs, freeze_resnet=True)

    # dispaly model summary
    model.summary()

    adam = Adam(lr=0.003)
    model.compile(optimizer=adam,loss='mse', metrics=['mean_squared_error'])

    # Callbacks: (1) Checkpoint
    checkpoint = ModelCheckpoint('check_point.h5', monitor='val_loss',save_best_only=True,
                                save_weights_only=False, mode='min')
    callbacks = [checkpoint]

    # ------TRAINING----------

    #if data_augmentation:
        #print('\WITH DATA AUGMENTATION')
    #    datagen = ImageDataGenerator(
    #        featurewise_center=False,  # set input mean to 0 over the dataset
    #        samplewise_center=False,  # set each sample mean to 0
    #        featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #        samplewise_std_normalization=False,  # divide each input by its std
    #        zca_whitening=False,  # apply ZCA whitening
    #        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    #        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #        horizontal_flip=True,  # randomly flip images
    #        vertical_flip=False)  # randomly flip images
    #else:
    #    print('WITHOUT DATA AUGMENTATION')

    model.fit(x_train,y_train, batch_size=batch_size, validation_data=(x_test,y_test),
                callbacks=callbacks, shuffle=True, epochs=epochs)

    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    print("\nloss:",score[0])
    print("\nmse:", score[1])

if __name__ == '__main__':
    train()
