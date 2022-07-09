import cv2
import time
import os
import sys
import argparse
from pathlib import Path
from screeninfo import get_monitors

from ImageProcessing import FrameProcessor, ProcessingVariables
from DisplayUtils.TileDisplay import show_img, reset_tiles
import headless

window_name = 'Playground (Esc quits)'
version = '_2_0'

d = headless.d

def get_arg_parser():
    parser = headless.get_arg_parser()
    return parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    headless.setup(args.file, args.script_dir)
    setup_ui()
    process_image()
    cv2.waitKey()

def process_image():
    reset_tiles()
    start_time = time.time()

    debug_images, output = headless.frameProcessor.process_image(d, d['iterations'])

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

    cv2.imshow(window_name, headless.frameProcessor.img)
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

def maintain_odd(x):
    if x % 2 == 0:
        x += 1
    return x

def change_blur(x):
    x = maintain_odd(x)
    change(x, 'blur')

def change_adj(x):
    change(x, 'adjustment')

def change_erode(x):
    change(x, 'erode')

def change_iterations(x):
    change(x, 'iterations')

def change_threshold(x):
    x = maintain_odd(x)
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
