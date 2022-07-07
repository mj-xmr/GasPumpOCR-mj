import cv2
import numpy as np
import os
import sys
from pathlib import Path
#from ImageProcessing.OpenCVUtils import inverse_colors, sort_contours

def get_updated_params(params_dict):
    params_dict['erode'] = 1
    params_dict['angle_degrees'] = 3

    return params_dict

def get_min_size_rectangle_one():
    return 10 * 11

def get_min_size_rectangle_():
    return 10 * 20

def get_desired_aspect_digit():
    return 0.46

def get_desired_aspect_digit_one():
    return 0.26

def get_final_number_multiplier():
    return 100

def rotate_image(image, angle):
    if angle == 0:
        return image
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def test():
    print("test_filter_module")
    
def test2():
    print("Test")
    HOME = str(Path.home()) + "/"
    
    file_name = HOME + "/devel/github/LCD-OCR-mj/Images/lcd3.png"
    file_name = HOME + "/devel/ocr_pic.jpg"
    #win = cv2.namedWindow("test")
    image_original = cv2.imread(file_name)

    lower_green = np.array([0, 0, 244])
    upper_green = np.array([255, 255,255])
    
    hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    res = cv2.bitwise_and(image_original, image_original, mask=mask)

    #res = rotate_image(res, 30)

    #cv2.imshow('orig',image_original)
    cv2.imshow('color mask', res)
    #process_image()
    cv2.waitKey(7)
    print("Post")
    # TODO: Cloee all windows?
    cv2.waitKey()
    
def get_debug_images(image_original, params_dict, iterations):
    d = params_dict
    return get_debug_images_new (image_original, params_dict, iterations)
    return get_debug_images_orig(image_original, d['blur'], d['threshold'], d['adjustment'], d['erode'], iterations)

def get_debug_images_new(image_original, params_dict, iterations):
    from ImageProcessing.OpenCVUtils import inverse_colors, sort_contours
    debug_images = []

    d = params_dict
    
    img = image_original.copy()
    debug_images.append(('Original', image_original))

    
    rotated = rotate_image(image_original, d['angle_degrees'])

    # Adjust the exposure
    alpha = d['exposure']
    exposure_img = cv2.multiply(rotated, np.array([alpha]))
    debug_images.append(('Exposure Adjust', exposure_img))

    hsv = cv2.cvtColor(exposure_img, cv2.COLOR_BGR2HSV)
    #https://stackoverflow.com/questions/47483951/how-to-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-image
    #, blur, threshold, adjustment, erode,
    loH = d['loH']
    loS = d['loS']
    loV = d['loV']
    hiH = d['hiH']
    hiS = d['hiS']
    hiV = d['hiV']
    lower_hsv = np.array([loH, loS, loV])
    upper_hsv = np.array([hiH, hiS, hiV])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    res = cv2.bitwise_and(exposure_img, exposure_img, mask=mask)
    debug_images.append(('Color mask', res))
    
    # Convert to grayscale
    img2gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    debug_images.append(('Grayscale', img2gray))

    # Blur to reduce noise
    blur = d['blur']
    img_blurred = cv2.GaussianBlur(img2gray, (blur, blur), 0)
    debug_images.append(('Blurred', img_blurred))

    cropped = img_blurred

    # Needed for my LCD
    cropped = inverse_colors(cropped)
    # Threshold the image
    cropped_threshold = cv2.adaptiveThreshold(cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              d['threshold'], d['adjustment'])
    debug_images.append(('Cropped Threshold', cropped_threshold))

    # Erode the lcd digits to make them continuous for easier contouring
    erode = d['erode']
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
    eroded = cv2.erode(cropped_threshold, kernel, iterations=iterations)
    debug_images.append(('Eroded', eroded))

    # Reverse the image to so the white text is found when looking for the contours
    inverse = inverse_colors(eroded)
    debug_images.append(('Inversed', inverse))
    
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

if __name__ == "__main__":
    test2()
