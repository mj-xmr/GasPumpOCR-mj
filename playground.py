import cv2
import time
import os
import sys
import argparse
from pathlib import Path
from screeninfo import get_monitors

from ImageProcessing import FrameProcessor, ProcessingVariables
from DisplayUtils.TileDisplay import show_img, reset_tiles

window_name = 'Playground (Esc quits)'
file_name = 'tests/single_line/49A95.jpg'
version = '_2_0'
HOME = str(Path.home()) + "/"

erode = ProcessingVariables.erode
threshold = ProcessingVariables.threshold
adjustment = ProcessingVariables.adjustment
iterations = ProcessingVariables.iterations
blur = ProcessingVariables.blur

std_height = 90

frameProcessor = FrameProcessor(std_height, version, True)

def GetParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--script-dir', default=HOME,  type=str, help="Script with a special order of filters")
    parser.add_argument('-f', '--file',   default=file_name, type=str, help="Image file to process")

    return parser

def handle_custom_script_file(script_dir):
    if script_dir:
        path_expected = script_dir + '/ocr_filter_module.py'
        if not os.path.isfile(path_expected):
            print("Not found OCR filter module:", path_expected, ". Using the default implementation.")
            pass
        else:
            print("Using filter module:", path_expected)
            sys.path.append(script_dir)
            import ocr_filter_module
            ocr_filter_module.test()
            frameProcessor.set_filter_module(ocr_filter_module)

def main():
    parser = GetParser()
    args = parser.parse_args()
    handle_custom_script_file(args.script_dir)

    img_file = args.file
    frameProcessor.set_image(img_file)
    setup_ui()
    process_image()
    cv2.waitKey()

def process_image():
    reset_tiles()
    start_time = time.time()
    debug_images, output = frameProcessor.process_image(blur, threshold, adjustment, erode, iterations)

    for image in debug_images:
        show_img(image[0], image[1])

    print("Processed image in %s seconds" % (time.time() - start_time))

    monitor0 = get_monitors()[0]
    screen_h = monitor0.height
    screen_w = monitor0.width
    print("Screen w/h =", screen_w, screen_h)
    window_x = 800
    window_y = 600
    if screen_h / 2 < window_y:
        window_y -= 200

    cv2.imshow(window_name, frameProcessor.img)
    cv2.moveWindow(window_name, window_x, window_y)


def setup_ui():
    cv2.namedWindow(window_name)
    cv2.createTrackbar('Threshold', window_name, int(threshold), 500, change_threshold)
    cv2.createTrackbar('Iterations', window_name, int(iterations), 5, change_iterations)
    cv2.createTrackbar('Adjust', window_name, int(adjustment), 200, change_adj)
    cv2.createTrackbar('Erode', window_name, int(erode), 5, change_erode)
    cv2.createTrackbar('Blur', window_name, int(blur), 25, change_blur)


def change_blur(x):
    global blur
    print('Adjust: ' + str(x))
    if x % 2 == 0:
        x += 1
    blur = x
    process_image()


def change_adj(x):
    global adjustment
    print('Adjust: ' + str(x))
    adjustment = x
    process_image()


def change_erode(x):
    global erode
    print('Erode: ' + str(x))
    erode = x
    process_image()


def change_iterations(x):
    print('Iterations: ' + str(x))
    global iterations
    iterations = x
    process_image()


def change_threshold(x):
    print('Threshold: ' + str(x))
    global threshold

    if x % 2 == 0:
        x += 1
    threshold = x
    process_image()


if __name__ == "__main__":
    main()
