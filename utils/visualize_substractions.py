import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def visualize_substractions(inputfile):
    input = cv.imread(inputfile)
    for i in range(11):
        template = cv.imread('assets/normal_dice/avg_'+str(i)+'.jpg')
        img = cv.subtract(input, template)
        ret, thresh = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
        plt.axis('off')
        plt.imshow(thresh)
        plt.show()


visualize_substractions(inputfile='assets/anomalous_dice/img_17433_cropped.jpg')
#visualize_substractions(inputfile='assets/normal_dice/3/272.jpg')
