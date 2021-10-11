import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def visualize_substractions(inputfile, plot: bool = False):
    """
    From an input picture (1st parameter) applies some transformation and comparisons to the set of 11 normal dice
    (templates). The comparisons consist in differences or subtractions so that the remaining pixels should account for
    the matching between input file and the templates: if the matching is good, few pixels, more pixels otherwise.
    :param inputfile: complete path to the input picture
    :param plot: whether to show a plot of the input and the transformations or not
    :return: two metrics result and result2 that account for the sum of remaining pixels after 2 sets of comparison
    """
    inputpic = cv.imread(inputfile)  # input picture with same size as dataset
    name = os.path.basename(inputfile).split('.')[0]  # name of file

    adjusted1 = cv.convertScaleAbs(inputpic, alpha=2.5, beta=-100)  # having more contrast

    if plot:  # subplots of the input and the transformations
        fig = plt.figure()
        fig.suptitle('Input '+name+' and subtraction + thresholding')
        plt.subplot(7, 4, 1), plt.title('Input'), plt.imshow(inputpic)

    result = []
    result2 = []

    for i in range(11):  # applying transformations with each of the 11 classes of normal dice
        template = cv.imread('assets/normal_dice/avg_'+str(i)+'.jpg')
        img = cv.absdiff(template, inputpic)
        img2 = cv.subtract(adjusted1, template)
        ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        ret, thresh2 = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)
        if plot:
            plt.axis('off')
            plt.subplot(7, 4, i+5), plt.title('template '+str(i), fontsize=8), plt.imshow(thresh)
            plt.subplot(7, 4, i+12+5), plt.title('template '+str(i), fontsize=8), plt.imshow(thresh2)
        result.append(np.sum(thresh))
        result2.append(np.sum(thresh2))

    if plot:
        plt.show()

    # minimum relative sum of pixels amongst the 11 transformations
    min_relative_result = min(result/max(result))
    min_relative_result2 = min(result2/max(result2))

    return min_relative_result, min_relative_result2


'''print(visualize_substractions(inputfile='assets/normal_dice_20211008/0/img_00923_cropped.jpg', plot=True))
visualize_substractions(inputfile='assets/normal_dice/6/700.jpg', plot=True)
visualize_substractions(inputfile='assets/anomalous_dice/img_17516_cropped.jpg', plot=True)
visualize_substractions(inputfile='assets/anomalous_dice/img_17433_cropped.jpg', plot=True)
visualize_substractions(inputfile='assets/anomalous_dice/img_17480_cropped.jpg', plot=True)
visualize_substractions(inputfile='assets/anomalous_dice/img_17583_cropped.jpg', plot=True)
visualize_substractions(inputfile='assets/anomalous_dice/img_17584_cropped.jpg', plot=True)
visualize_substractions(inputfile='assets/normal_dice/3/272.jpg', plot=True)
visualize_substractions(inputfile='assets/normal_dice/6/777.jpg', plot=True)
visualize_substractions(inputfile='assets/normal_dice/10/1068.jpg', plot=True)'''
