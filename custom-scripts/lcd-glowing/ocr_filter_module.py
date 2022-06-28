import cv2
import numpy as np
import os
import sys
from pathlib import Path

#from ImageProcessing.OpenCVUtils import inverse_colors, sort_contours

def test():
    print("test_filter_module")
    
def test2():
    print("Test")
    HOME = str(Path.home()) + "/"
    
    file_name = HOME + "/devel/github/LCD-OCR-mj/Images/lcd3.png"
    file_name = HOME + "/devel/ocr_pic.jpg"
    #win = cv2.namedWindow("test")
    image_original = cv2.imread(file_name)

    
    hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,0,0])
    upper_red = np.array([0,255,255])
    upper_red = np.array([255, 255, 240 ])

    mask = cv2.inRange(hsv, lower_red, upper_red)

    res = cv2.bitwise_and(image_original,image_original, mask= mask)

    
    #cv2.imshow('orig',image_original)
    cv2.imshow('color mask',res)
    #process_image()
    cv2.waitKey(7)
    print("Post")
    # TODO: Cloee all windows?
    cv2.waitKey()
    
def get_debug_images(image_original, blur, threshold, adjustment, erode, iterations):
    return get_debug_images_new (image_original, blur, threshold, adjustment, erode, iterations)
    return get_debug_images_orig(image_original, blur, threshold, adjustment, erode, iterations)
    
def get_debug_images_new(image_original, blur, threshold, adjustment, erode, iterations):    
    from ImageProcessing.OpenCVUtils import inverse_colors, sort_contours
    debug_images = []

    alpha = float(2.5)

    img = image_original.copy()

    debug_images.append(('Original', image_original))

    hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
    #lower_red = np.array([0,0,0])
    #upper_red = np.array([0,255,255])
    #https://stackoverflow.com/questions/47483951/how-to-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-image
    lower_red = np.array([40, 40,40])
    upper_red = np.array([70, 255,255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    res = cv2.bitwise_and(image_original,image_original, mask= mask)
    debug_images.append(('Color mask', res))
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
