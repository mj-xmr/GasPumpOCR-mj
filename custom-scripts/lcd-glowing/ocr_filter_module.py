import cv2
import numpy as np
import os

from ImageProcessing.OpenCVUtils import inverse_colors, sort_contours

def test():
    print("test_filter_module")

def get_debug_images(image_original, blur, threshold, adjustment, erode, iterations):
    return get_debug_images_orig(image_original, blur, threshold, adjustment, erode, iterations)
    return get_debug_images_new (image_original, blur, threshold, adjustment, erode, iterations)
    
def get_debug_images_new(image_original, blur, threshold, adjustment, erode, iterations):    
    debug_images = []

    inverse = inverse_colors(image_original)
    debug_images.append(('Inversed', inverse))

    debug_images.append(image_original)
    
    return debug_images


def get_debug_images_orig(image_original, blur, threshold, adjustment, erode, iterations):
    debug_images = []
    alpha = float(2.5)

    img = image_original.copy()

    debug_images.append(('Original', image_original))

    # Adjust the exposure
    exposure_img = cv2.multiply(img, np.array([alpha]))
    debug_images.append(('Exposure Adjust', exposure_img))

    # Convert to grayscale
    img2gray = cv2.cvtColor(exposure_img, cv2.COLOR_BGR2GRAY)
    debug_images.append(('Grayscale', img2gray))

    # Blur to reduce noise
    img_blurred = cv2.GaussianBlur(img2gray, (blur, blur), 0)
    debug_images.append(('Blurred', img_blurred))

    cropped = img_blurred

    # Threshold the image
    cropped_threshold = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              threshold, adjustment)
    debug_images.append(('Cropped Threshold', cropped_threshold))

    # Erode the lcd digits to make them continuous for easier contouring
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
    eroded = cv2.erode(cropped_threshold, kernel, iterations=iterations)
    debug_images.append(('Eroded', eroded))

    # Reverse the image to so the white text is found when looking for the contours
    inverse = inverse_colors(eroded)
    debug_images.append(('Inversed', inverse))

    return debug_images
