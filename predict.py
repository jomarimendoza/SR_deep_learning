import os
from glob import glob
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from h5py import File as h5file

import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import floor,ceil
from plotting import display_multi_images, plot_img

import models
from keras.layers import Input

""" Resize image with 3 channels """
def resize_3channels(image, height, width, interpolation=cv2.INTER_CUBIC):
    image_resize = np.zeros((height, width,3))

    Ch0 = cv2.resize(image[:,:,0], (width,height), interpolation=interpolation)
    Ch1 = cv2.resize(image[:,:,1], (width,height), interpolation=interpolation)
    Ch2 = cv2.resize(image[:,:,2], (width,height), interpolation=interpolation)

    image_resize[:,:,0] = Ch0
    image_resize[:,:,1] = Ch1
    image_resize[:,:,2] = Ch2

    return image_resize

""" Reconstruct colored image"""
def recon_image(img, model_type='ees'):
    scale = 4

    # Load image from data set
    hr_img = cv2.imread(img, cv2.IMREAD_COLOR)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)

    # 4x4 divisibility dimensions
    height = int( scale * floor( hr_img.shape[0] / scale ))
    width = int( scale * floor( hr_img.shape[1] / scale ))

    # Resize and store to new YCbCr image
    hr_img = resize_3channels(hr_img,height,width)
    hr_img = hr_img.astype('uint8')

    # Get low res image of intensity component (Y)
    lr_Y = cv2.resize(hr_img[:,:,0], (0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)

    # Set number of patches to be made
    height = lr_Y.shape[0]
    width = lr_Y.shape[1]
    HEIGHT = height*scale
    WIDTH = width*scale

    inputs = Input(shape=(height, width, 1))
    if(model_type == 'ees'):
        model = models.EES_model(inputs)
        model.load_weights('ees_weights.h5')
    elif(model_type == 'eed'):
        model = models.EED_model(inputs)
        model.load_weights('eed_weights.h5')
    elif(model_type == 'eeds'):
        model = models.EEDS_model(inputs)
        model.load_weights('eeds_weights.h5')

    # Reshape input to fit model
    lr_Y = np.reshape(lr_Y,(1,height, width,1))
    lr_Y = lr_Y.astype('float32') / 255
    hr_Y = model.predict(lr_Y)*255
    hr_Y = np.reshape(hr_Y, (HEIGHT, WIDTH) )

    # Change Y channel with reconstructed
    # Preserve CrCb
    hr_img[:,:,0] = hr_Y

    recon_img = cv2.cvtColor(hr_img, cv2.COLOR_YCrCb2BGR)
    return recon_img

if __name__ == '__main__':
    """ CHANGE MODEL HERE """
    img = recon_image('./Set14/img_001_SRF_4_HR.png', model_type='ees')
    plot_img(img, title='Sample image', with_color=True)

    # Read an image
    imageSet = glob('./Set14/*')

    images = []
    titles = []
    for img in imageSet:
        images.append(recon_image(img))
        title = 'Image ' + img.split('_')[1]
        titles.append(title)
    # 2x7 subplot
    display_multi_images(images, 3, 5, titles, with_color=True)
