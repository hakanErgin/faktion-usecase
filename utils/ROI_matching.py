import cv2
import numpy as np
import glob


anomalous_images = [cv2.imread(file, 0) for file in glob.glob("./assets/anomalous_dice/*.jpg")]


templates = []
for idx in range(11):
    templates.append(cv2.imread("./assets/avg_normal_dice/avg_" + str(idx) + ".jpg", 0))


# mask corners with a circle
def mask_corners(image):
    circle = np.zeros((128, 128), dtype="uint8")
    cv2.circle(circle, (64, 64), 60, 255, -1)
    masked_image = cv2.bitwise_and(image, circle)
    masked_image[masked_image == 0] = 255

    # concatenated_images = cv2.hconcat([image, circle, masked_image])
    return masked_image


# otsu if needed
otsu = cv2.THRESH_BINARY | cv2.THRESH_OTSU

# apply (hardcoded) thresholding value


def apply_thresholding(masked_image, threshold=cv2.THRESH_BINARY):
    (T, thresh_image) = cv2.threshold(masked_image, 100 if threshold == cv2.THRESH_BINARY else 0, 255,
                                      threshold)
    if not threshold == cv2.THRESH_BINARY:
        print(T)
    return thresh_image


def apply_preprocessing(input):
    corners_masked_image = mask_corners(input)
    thresholded_image = apply_thresholding(corners_masked_image)
    return thresholded_image

# thresholded_image, corners_masked_image = apply_preprocessing(input_image)


# indexes of black dots or features
def get_roi(thresholded_image):
    template_roi = np.argwhere(thresholded_image == 0)
    return template_roi


bw_templates = []
for template in templates[-11:]:
    thresholded_image = apply_preprocessing(template)
    bw_templates.append(apply_thresholding(thresholded_image))
    # print(thresholded_image.shape)

len(bw_templates)


bw_anomalous_images = []
for anomalous_image in anomalous_images:
    thresholded_image = apply_preprocessing(anomalous_image)
    bw_anomalous_images.append(apply_thresholding(thresholded_image))
    # print(thresholded_image.shape)

len(bw_anomalous_images)


# in order to find the classes
def compare_images(bw_templates, bw_anomalous_image):
    comparisons = []
    comparisons.append(bw_anomalous_image)
    for bw_template in bw_templates:
        comparisons.append(cv2.subtract(bw_template, bw_anomalous_image))
    return comparisons


# comparisons = compare_images(bw_templates, bw_anomalous_images[82])
comparisons = compare_images(bw_templates, bw_anomalous_images[9])
cv2.imshow('', cv2.hconcat(comparisons))
# cv2.imshow('', cv2.subtract(anomalous_image, template_0))
cv2.waitKey()
cv2.destroyAllWindows()
