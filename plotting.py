import cv2
import numpy as np
from matplotlib import pyplot as plt

""" Dispalys a loaded image to numpy """
def plot_img(img, title='Image', with_color=False):
    if(with_color):
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap='gray')

    plt.title(title)
    plt.axis('off')
    print("Exit image")
    plt.show()
    plt.close()


""" Displays multiple images """
def display_multi_images(images, row, col, titles, suptitle='Images', with_color=False):
    fig=plt.figure(figsize=(8, 8))
    fig.suptitle(suptitle,fontsize=14)

    for i in range(1, col*row +1):
        fig.add_subplot(row, col, i)
        if(with_color):
            plt.imshow(cv2.cvtColor(images[i-1], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i-1], cmap='gray')
        plt.title(titles[i-1])
        plt.axis('off')

        if(i >= len(images)):
            break

    plt.axis('off')
    print("Exit image")
    plt.show()
    plt.close()

if __name__ == '__main__':
    print('For image plotting')
