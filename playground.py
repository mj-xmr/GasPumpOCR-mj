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

d = {}
d['exposure'] = ProcessingVariables.exposure
d['blur'] = ProcessingVariables.blur
d['erode'] = ProcessingVariables.erode
d['threshold'] = ProcessingVariables.threshold
d['adjustment'] = ProcessingVariables.adjustment
d['iterations'] = ProcessingVariables.iterations

d['loH'] = ProcessingVariables.loH
d['loS'] = ProcessingVariables.loS
d['loV'] = ProcessingVariables.loV
d['hiH'] = ProcessingVariables.hiH
d['hiS'] = ProcessingVariables.hiS
d['hiV'] = ProcessingVariables.hiV
    
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
            global d
            d = frameProcessor.filter_module.get_updated_params(d)

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

    debug_images, output = frameProcessor.process_image(d, d['iterations'])

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
        window_y -= 300

    cv2.imshow(window_name, frameProcessor.img)
    cv2.moveWindow(window_name, window_x, window_y)


def setup_ui():
    cv2.namedWindow(window_name)
    cv2.createTrackbar('loH', window_name, int(d['loH']), 255, change_loH)
    cv2.createTrackbar('loS', window_name, int(d['loS']), 255, change_loS)
    cv2.createTrackbar('loV', window_name, int(d['loV']), 255, change_loV)
    cv2.createTrackbar('hiH', window_name, int(d['hiH']), 255, change_hiH)
    cv2.createTrackbar('hiS', window_name, int(d['hiS']), 255, change_hiS)
    cv2.createTrackbar('hiV', window_name, int(d['hiV']), 255, change_hiV)
    cv2.createTrackbar('Exposure', window_name, int(d['exposure'] * 100), 500, change_exposure)
    cv2.createTrackbar('Blur', window_name, int(d['blur']), 25, change_blur)
    cv2.createTrackbar('Threshold', window_name, int(d['threshold']), 500, change_threshold)
    cv2.createTrackbar('Adjust', window_name, int(d['adjustment']), 200, change_adj)
    cv2.createTrackbar('Iterations', window_name, int(d['iterations']), 5, change_iterations)
    cv2.createTrackbar('Erode', window_name, int(d['erode']), 5, change_erode)

def change(xx, var_name):
    print(var_name + ":", xx)
    d[var_name] = xx
    process_image()

def change_exposure(x):
    change(x/100, 'exposure')

def change_blur(x):
    if x % 2 == 0:
        x += 1
    change(x, 'blur')

def change_adj(x):
    change(x, 'adjustment')

def change_erode(x):
    change(x, 'erode')

def change_iterations(x):
    change(x, 'iterations')

def change_threshold(x):
    if x % 2 == 0:
        x += 1
    change(x, 'threshold')

def change_loH(x):
    change(x, 'loH')

def change_loS(x):
    change(x, 'loS')

def change_loV(x):
    change(x, 'loV')

def change_hiH(x):
    change(x, 'hiH')

def change_hiS(x):
    change(x, 'hiS')

def change_hiV(x):
    change(x, 'hiV')

if __name__ == "__main__":
    main()
