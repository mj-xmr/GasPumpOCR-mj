import cv2
import numpy as np
import os
import sys
from pathlib import Path

def get_updated_params(params_dict):
    return params_dict

def get_min_size_rectangle_one():
    return 100 * 10

def get_min_size_rectangle_():
    return 100 * 20

def get_desired_aspect_digit():
    return 0.6

def get_desired_aspect_digit_one():
    return 0.3

def get_final_number_multiplier():
    return 1

def test():
    print("test OCR filter_module gas pump")
    
def get_debug_images(image_original, params_dict, iterations, scaling_factors):
    d = params_dict
    return get_debug_images_orig(image_original, d['blur'], d['threshold'], d['adjustment'], d['erode'], iterations)

def get_debug_images_orig(image_original, blur, threshold, adjustment, erode, iterations):
    from ImageProcessing.OpenCVUtils import inverse_colors, mask_image_rect, mask_image_hsv_dict, rotate_image_simple
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

