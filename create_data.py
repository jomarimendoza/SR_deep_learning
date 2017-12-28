import os
from glob import glob
import cv2
import numpy as np
from h5py import File as h5file
from matplotlib import pyplot as plt

"""Save image data and its label(s) to an hdf5 file."""
def save_data_hdf5(data1, data2, filename):
    x = data1.astype('float32')
    y = data2.astype('float32')
    h5_filename = filename + '.h5'

    with h5file(h5_filename,'w') as dataset:
        dataset.create_dataset('data', data=x, shape=x.shape)
        dataset.create_dataset('label', data=y, shape=y.shape)

""" Dispalys a loaded image to numpy """
def plot_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.close()
    print(img.shape)

""" Generate data set """
def generate_data(_path, scale=4, count_files=10):
    patch_size = 96
    num_random_patches = 30

    low_res = []
    high_res = []

    # to list path of images (train/test)
    _files = _path + '/*'
    imageSet = glob(_files)


    for idx in range(count_files):
        print('Processing ',imageSet[idx],'...')

        # Read image
        hr_img = cv2.imread(imageSet[idx], cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:,:,0] # INTENSITY VALUE ONLY

        # Number of patches (num_files x 30)
        # random coordinates within the image
        x_points = np.random.randint(0, min(hr_img.shape[0],hr_img.shape[1]) - patch_size, size=num_random_patches)
        y_points = np.random.randint(0, min(hr_img.shape[0],hr_img.shape[1]) - patch_size, size=num_random_patches)

        for x,y in zip(x_points,y_points):
            # Get all patches from LR and HR image
            hr_patch = hr_img[x:x + patch_size, y:y + patch_size]
            lr_patch = cv2.resize(hr_patch, (0,0), fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)

            # Normalize the pixels
            hr_patch = hr_patch.astype('float32') / 255
            lr_patch = lr_patch.astype('float32') / 255

            # Store patches
            low_res.append(lr_patch)
            high_res.append(hr_patch)

    print(len(low_res))
    # Stack the numpy array, at first axis for batch processing
    low_res = np.stack(low_res, axis=0)
    high_res = np.stack(high_res, axis=0)
    return low_res, high_res

if __name__ == '__main__':
    TRAIN_PATH = './DIV2K_train_HR'
    VALID_PATH = './DIV2K_valid_HR'

    # Div 2k files
    num_files_train = 800
    num_files_valid = 100
    scale = 4

    low_train, high_train = generate_data(TRAIN_PATH, scale=scale, count_files=num_files_train)
    low_valid, high_valid = generate_data(VALID_PATH, scale=scale, count_files=num_files_valid)

    save_data_hdf5(low_train, high_train,'train')
    save_data_hdf5(low_valid, high_valid,'valid')

    # Plot patches (training)
    #for i in range(low_train.shape[0]):
    #    plot_image(low_train[i,:,:])
    #    plot_image(high_train[i,:,:])
