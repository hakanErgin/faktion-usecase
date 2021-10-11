import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Open the image with openCV
template = cv.imread('assets/normal_dice/2/122.jpg')
template = cv.imread('assets/normal_dice/3/272.jpg')
template = cv.imread('assets/normal_dice/avg_0.jpg')
input = cv.imread('assets/anomalous_dice/img_17480_cropped.jpg')
# input = cv.imread('assets/normal_dice/1/157.jpg')

img = cv.subtract(template, input)

# Display the images with openCV
plt.axis('off')
plt.imshow(template)
plt.show()
plt.imshow(input)
plt.show()
plt.imshow(img)
plt.show()

ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
plt.imshow(thresh1)
plt.show()

numpydata = np.asarray(img)
# <class 'numpy.ndarray'>
print(type(numpydata))
print(type(img))
#  shape
print(numpydata.shape)
print(img.shape)
