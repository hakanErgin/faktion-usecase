import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Open the image with openCV
#template = cv.imread('assets/normal_dice/2/122.jpg')
#template = cv.imread('assets/normal_dice/3/272.jpg')
template = cv.imread('assets/normal_dice/avg_1.jpg', -1)
input = cv.imread('assets/anomalous_dice/img_17480_cropped.jpg', -1)
#input = cv.imread('assets/normal_dice/1/157.jpg', -1)

print(input.shape)

def grayscaleHistogram(input):
    plt.subplot(1, 2, 1), plt.imshow(input)
    #plt.show()
    # compute a grayscale histogram
    hist = cv.calcHist([input], [0], None, [256], [0, 256])
    # plot the histogram
    plt.subplot(1, 2, 2)
    #plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()


img = cv.subtract(template, input)

'''# Display the images with openCV
plt.axis('off')
plt.imshow(template)
plt.show()
plt.imshow(input)
plt.show()
plt.imshow(img)
plt.show()'''

ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
#plt.imshow(thresh1)
#plt.show()

'''numpydata = np.asarray(img)
# <class 'numpy.ndarray'>
print(type(numpydata))
print(type(img))
#  shape
print(numpydata.shape)
print(img.shape)
'''


#grayscaleHistogram(template)
#grayscaleHistogram(input)
#grayscaleHistogram(img)
#grayscaleHistogram(thresh1)



img1 = cv.imread('assets/anomalous_dice/img_18224_cropped.jpg')
#img1 = cv.imread('assets/normal_dice/avg_1.jpg')
#img1 = cv.imread('assets/normal_dice/1/157.jpg')
adjusted1 = cv.convertScaleAbs(img1, alpha=2, beta=0)
cv.imshow('original', img1)
cv.imshow('adjusted', adjusted1)
cv.waitKey()

