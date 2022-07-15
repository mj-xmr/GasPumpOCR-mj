import cv2
import time
import os
import sys
import argparse
from pathlib import Path
import importlib

from ImageProcessing import FrameProcessor, ProcessingVariables

file_name = 'tests/single_line/49A95.jpg'
version = '_2_0'
HOME = str(Path.home()) + "/"

d = {}
std_height = 90
frameProcessor = None

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--script-dir', default=HOME,  type=str, help="Script with a special order of filters")
    parser.add_argument('-f', '--file',   default=file_name, type=str, help="Image file to process")

    return parser

def init_params():
    global d
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
    
def handle_custom_script_file(script_dir):
    if script_dir:
        path_expected = script_dir + '/ocr_filter_module.py'
        if not os.path.isfile(path_expected):
            print("Not found OCR filter module:", path_expected, ". Using the default implementation.")
            pass
        else:
            print("Using filter module:", path_expected)
            sys.path.insert(0, script_dir)
            import ocr_filter_module # For this reason, the script filename must be fixed
            importlib.reload(ocr_filter_module)
            ocr_filter_module.test()
            frameProcessor.set_filter_module(ocr_filter_module)
            global d
            d = frameProcessor.filter_module.get_updated_params(d)

def setup(file, script_dir):
    global frameProcessor
    init_params()
    frameProcessor = FrameProcessor(std_height, version, True)
    handle_custom_script_file(script_dir)
    img_file = file
    frameProcessor.set_image(img_file)

def get_detection(file, script_dir):
    start_time = time.time()
    setup(file, script_dir)
    debug_images, output = frameProcessor.process_image(d, d['iterations'])
    print("Processed image in %s seconds" % (time.time() - start_time))
    return output

def main(file, script_dir):
    return get_detection(file, script_dir)

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args.file, args.script_dir)
