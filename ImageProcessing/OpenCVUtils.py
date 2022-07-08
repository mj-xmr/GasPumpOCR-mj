import cv2
import math
import numpy as np

# http://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c/33564950#33564950
def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=(255, 255, 255))
    return rotated_mat


def rotate_image_simple(image, angle):
    if angle == 0:
        return image
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def mask_image_rect(img, mask_xywh, scaling_factors, foreground=0, background=255):
    mask = np.zeros(img.shape[0:2], dtype='uint8')
    mask.fill(background)
    scaling_factor_x = scaling_factors[0]
    scaling_factor_y = scaling_factors[1]
    #print("Mask original:", rect)
    x = round(mask_xywh[0] * scaling_factor_x)
    y = round(mask_xywh[1] * scaling_factor_y)
    w = round(mask_xywh[2] * scaling_factor_x)
    h = round(mask_xywh[3] * scaling_factor_y)
    #print("Mask scaled:", x, y, w, h)
    mask[y:y+h,x:x+w] = foreground
    #print("Mask & image shape:", mask.shape, "&", img.shape)
    img = cv2.bitwise_and(img,img,mask=mask)
    return img

def mask_image_hsv(img, loH, loS, loV, hiH, hiS, hiV):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #https://stackoverflow.com/questions/47483951/how-to-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-image
    lower_hsv = np.array([loH, loS, loV])
    upper_hsv = np.array([hiH, hiS, hiV])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    masked = cv2.bitwise_and(img, img, mask=mask)

    return masked

def mask_image_hsv_dict(img, d):
    loH = d['loH']
    loS = d['loS']
    loV = d['loV']
    hiH = d['hiH']
    hiS = d['hiS']
    hiV = d['hiV']

    return mask_image_hsv(img, loH, loS, loV, hiH, hiS, hiV)

def inverse_colors(img):
    img = (255 - img)
    return img


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes),
                                         key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, bounding_boxes
