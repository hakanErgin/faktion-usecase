import cv2
import numpy as np
import glob
import random as rng


# reading the files to work with
anomalous_images = [cv2.imread(file, 0) for file in glob.glob("./assets/anomalous_dice/*.jpg")]
templates = []
for idx in range(11):
    templates.append(cv2.imread("./assets/normal_dice/avg_"+ str(idx) +".jpg", 0))

rng.seed(12345)
edge_threshold = 500


'''
Preprocessing
'''
def mask_corners(image):
    # 1-masks corners with a circular shape
    circle = np.zeros((128, 128), dtype = "uint8")
    cv2.circle(circle, (64, 64), 60, 255, -1)
    masked_image = cv2.bitwise_and(image, circle)
    masked_image[masked_image==0] = 255
    return masked_image

def apply_thresholding(masked_image, threshold=cv2.THRESH_BINARY):
    # 2-applies (hardcoded) thresholding value - removes all unnecessary details
    (T, thresh_image) = cv2.threshold(masked_image, 127, 255,
        threshold)
    return thresh_image

def draw_contours(image):
    # 3-find and draw contours
    # Detect edges using Canny
    canny_output = cv2.Canny(image, edge_threshold, edge_threshold * 2)
    # Find contours
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)

    return drawing

def apply_preprocessing_to_a_single_image(input):
# Applies the preprocessing functions created above to the input file
    corners_masked_image = mask_corners(input)
    thresholded_image = apply_thresholding(corners_masked_image)
    drawing_with_contours = draw_contours(thresholded_image)
    return drawing_with_contours

def apply_preprocessing_to_list(list):
    drawing_with_contours = []
    for image in list:
        thresholded_image = apply_preprocessing_to_a_single_image(image)
        drawing_with_contours.append(thresholded_image)
    return drawing_with_contours

# creating preprocessed images to work with
template_drawings_with_contours = apply_preprocessing_to_list(templates)

'''
finding the class of input from the templates
'''
def find_class_of_input(input):
    matches = []
    for i, template in enumerate(template_drawings_with_contours):
        result = cv2.matchTemplate(template, input, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # print(i+1, ' - ', max_val)
        matches.append({'class':i, 'confidence':max_val})
    max_match = max(matches, key=lambda x:x['confidence'])
    return max_match


def detect_anomaly():
    file = input("enter the path of a 128x128")
    image = cv2.imread(file, 0)
    drawing_with_contours = apply_preprocessing_to_a_single_image(image)

    # get class that matches in the form of dictionary - ex:{'class:': 5, 'confidence': 0.42129555344581604}
    input_class = find_class_of_input(drawing_with_contours)
    print(input_class, "\n")

    class_to_compare_with = input_class['class']

    # similarity between original templates and the input
    similarity_result = cv2.matchTemplate(templates[class_to_compare_with], 
    image, cv2.TM_CCOEFF_NORMED)[0][0]
    print('similarity between originals', similarity_result)

    normal_class_images = [cv2.imread(file, 0) for file in glob.glob("./assets/normal_dice/" + str(class_to_compare_with) + "/*.jpg")]

    similarity_results = []
    for normal_class_image in normal_class_images:
        similarity = cv2.matchTemplate(mask_corners(templates[class_to_compare_with]), 
        mask_corners(normal_class_image), cv2.TM_CCOEFF_NORMED)[0][0]
        similarity_results.append(similarity)

    print('min similarity between org template and input:\n', min(similarity_results), '\n')

    # visualizing the results
    list_to_plot = [drawing_with_contours, *template_drawings_with_contours]
    cv2.imshow('', cv2.hconcat(list_to_plot))
    cv2.waitKey()
    cv2.destroyAllWindows()

    # any input is normal unless similarity between originals is strictly lower than minimum similarity results
    if min(similarity_results) <= similarity_result:
        return 0
    else:
        return 1

    
detect_anomaly()


