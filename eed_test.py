import cv2
import numpy as np
from math import floor
from glob import glob
from predict import recon_image, resize_3channels
from compare_images import psnr
from ssim import ssim

def psnr_ssim(hr_img_name, recon_img):
    scale = 4

    # Load image from data set
    hr_img = cv2.imread(hr_img_name, cv2.IMREAD_COLOR)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
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

def evaluate(folder_path):
    imageSet = glob(folder_path + '/*')

    psnr_values = []
    ssim_values = []

    print("Processing images ")
    for i,img in enumerate(imageSet):
        recon_img = recon_image(img,model_type='eed')
        psnr_value, ssim_value = psnr_ssim(img,recon_img)

        psnr_values.append(psnr_value)
        ssim_values.append(np.mean(ssim_value))
        print(i+1,'/',len(imageSet), end="\r")

    psnr_ave = sum(psnr_values) / len(psnr_values)
    ssim_ave = sum(ssim_values) / len(ssim_values)

    return psnr_ave,ssim_ave

if __name__ == '__main__':
    psnr_ave, ssim_ave = evaluate('./Set14')
    print("PSNR: ", psnr_ave)
    print("SSIM: ", ssim_ave)
