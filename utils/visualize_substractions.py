import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def visualize_substractions(inputfile):
    inputpic = cv.imread(inputfile)
    name = os.path.basename(inputfile).split('.')[0]

    '''    fig = plt.figure()
    fig.suptitle('Input '+name+' and average templates')
    plt.subplot(4, 4, 1), plt.title('Input'), plt.imshow(inputpic)
    for i in range(11):
        template = cv.imread('assets/normal_dice/avg_'+str(i)+'.jpg')
        plt.axis('off')
        plt.subplot(4, 4, i+5), plt.title('template '+str(i)), plt.imshow(template)
    plt.show()
    '''

    #fig = plt.figure()
    #fig.suptitle('Input '+name+' and subtraction + thresholding')
    #plt.subplot(7, 4, 1), plt.title('Input'), plt.imshow(inputpic)
    result = []
    result2 = []
    for i in range(11):
        template = cv.imread('assets/normal_dice/avg_'+str(i)+'.jpg')
        img = cv.absdiff(template, inputpic)
        img2 = cv.subtract(inputpic, template)
        ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        ret, thresh2 = cv.threshold(img2, 25, 255, cv.THRESH_BINARY)
        #plt.axis('off')
        #plt.subplot(7, 4, i+5), plt.title('template '+str(i), fontsize=8), plt.imshow(img)
        #plt.subplot(7, 4, i+12+5), plt.title('template '+str(i), fontsize=8), plt.imshow(thresh2)
        result.append(np.sum(thresh))
        result2.append(np.sum(thresh2))
    sum_thresh = [sum(x) / 1e6 for x in zip(result, result2)]
    # print(sum_thresh)
    min_index = sum_thresh.index(min(sum_thresh))
    metric = max(result[min_index], result2[min_index])
    #print(f'Input {name} is from class {min_index} and the maximum sum of pixels after operations with this class is {metric}')
    #plt.show()
    result = min(result/max(result))
    return result


visualize_substractions(inputfile='assets/normal_dice/6/791.jpg')
visualize_substractions(inputfile='assets/anomalous_dice/img_17516_cropped.jpg')
visualize_substractions(inputfile='assets/anomalous_dice/img_17517_cropped.jpg')
visualize_substractions(inputfile='assets/anomalous_dice/img_17433_cropped.jpg')
visualize_substractions(inputfile='assets/anomalous_dice/img_17480_cropped.jpg')
visualize_substractions(inputfile='assets/anomalous_dice/img_17583_cropped.jpg')
visualize_substractions(inputfile='assets/anomalous_dice/img_17584_cropped.jpg')
visualize_substractions(inputfile='assets/normal_dice/3/272.jpg')
visualize_substractions(inputfile='assets/normal_dice/6/777.jpg')
visualize_substractions(inputfile='assets/normal_dice/10/1068.jpg')
