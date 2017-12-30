from glob import glob
import numpy as np
from predict import resize_3channels
import matplotlib.pyplot as plt
from math import floor,sqrt,log10
import cv2
from ssim import ssim
from plotting import display_multi_images, plot_img

""" FROM https://stackoverflow.com/questions/44944455/psnr-values-differ-in-matlab-implementation-and-python """
def psnr(target, ref):
    import cv2
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)

    diff = ref_data - target_data
    #print(diff.shape)
    diff = diff.flatten('C')

    rmse = sqrt(np.mean(diff ** 2.))

    return 20 * log10(255 / rmse)

""" measure psnr and ssim of two images """
def psnr_ssim(hr_img_name, recon_img_name):
    scale = 4

    # Load image from data set
    hr_img = cv2.imread(hr_img_name, cv2.IMREAD_COLOR)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
    recon_img = cv2.imread(recon_img_name, cv2.IMREAD_COLOR)
    recon_img = cv2.cvtColor(recon_img, cv2.COLOR_BGR2YCrCb)

    height = int( scale * floor( hr_img.shape[0] / scale ))
    width = int( scale * floor( hr_img.shape[1] / scale ))

    # Resize and store to new YCbCr image
    hr_img = resize_3channels(hr_img,height,width)
    recon_img = resize_3channels(recon_img,height,width)
    hr_img = hr_img.astype('uint8')
    recon_img = recon_img.astype('uint8')

    psnr_value = psnr(recon_img[:,:,0], hr_img[:,:,0])
    ssim_value = ssim(recon_img[:,:,0], hr_img[:,:,0])

    return psnr_value, ssim_value

""" Get average measurements of one method """
""" Created separate folders for different reconstructions """
def psnr_ssim_folder(folder_path):
    imageSet = glob(folder_path + '/*')

    """ Change per method """
    is_hr_first = False

    psnr_values = []
    ssim_values = []
    for i in range(len(imageSet)//2):
        idx = 2*i
        if(is_hr_first):
            hr_img = imageSet[idx]
            recon_img = imageSet[idx+1]
        else:
            recon_img = imageSet[idx]
            hr_img = imageSet[idx+1]

        psnr_value,ssim_value = psnr_ssim(hr_img, recon_img)
        psnr_values.append(psnr_value)
        ssim_values.append(np.mean(ssim_value))

    psnr_ave = sum(psnr_values) / len(psnr_values)
    ssim_ave = sum(ssim_values) / len(ssim_values)

    return psnr_ave, ssim_ave
if __name__ == '__main__':
    psnr_ave, ssim_ave = psnr_ssim_folder("./comparison_folder")

    print("PSNR: ", psnr_ave)
    print("SSIM: ", ssim_ave)

    #fig=plt.figure(figsize=(8, 8))
    #fig.suptitle("Title centered above all subplots", fontsize=14)
