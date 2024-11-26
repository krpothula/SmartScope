import os
import sys
import cv2
from typing import Dict
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detectors'))

from .detectors.detect_squares import detect
from .detectors.detect_holes import detect_holes, detect_and_classify_holes
from ..basic_finders import find_square_center
import logging
import torch

from Smartscope.lib.image_manipulations import fourier_crop
from Smartscope.lib.image.montage import Montage
logger = logging.getLogger(__name__)

WEIGHT_DIR = os.path.join(os.getenv("TEMPLATE_FILES"), 'weights')
IS_CUDA = False if eval(os.getenv('FORCE_CPU')) else torch.cuda.is_available() 


def find_squares(montage, **kwargs):
    logger.info('Running AI find_squares')
    kwargs['weights'] = os.path.join(WEIGHT_DIR, kwargs['weights'])
    if not IS_CUDA:
        kwargs['device'] = 'cpu'
    squares, labels, _, _ = detect(montage.image, **kwargs)
    success = True
    if len(squares) < 20 and montage.image.shape[0] > 20000:
        success = False
    logger.info(f'AI square finder found {len(squares)} squares')
    logger.debug(f'{squares},{type(squares)}')
    squares = [i.numpy() for i in squares]
    return (squares, labels), success, dict()


def find_holes(montage:Montage, class_map:Dict=None, success_threshold:int=10,  **kwargs):

    def filter_hole_class(hole):
        return class_map[hole[-1]].name == 'Hole'
    
    logger.info('Running AI hole detection')
    # centroid = find_square_center(montage.image)
    kwargs['weights_circle'] = os.path.join(WEIGHT_DIR, kwargs['weights_circle'])
    if not IS_CUDA:
        kwargs['device'] = 'cpu'
    image = montage.image
    binning=1
    # if montage.pixel_size < 100:
    #     logger.info(f'Resizing image')
    #     binning = (150/montage.pixel_size)
    #     image = fourier_crop(image, height=montage.shape_x/binning)
    #     pad_x = int((montage.shape_x - image.shape[0]) //2)
    #     pad_y = int((montage.shape_y - image.shape[1]) //2)
    #     image = cv2.copyMakeBorder(image,pad_x,pad_x,pad_y,pad_y,cv2.BORDER_CONSTANT,value=0)
    logger.debug(f'Resized shape: {image.shape}, original: {montage.image.shape}')
    all_targets = detect_holes(image, **kwargs)
    holes = list(filter(lambda x: filter_hole_class(x), all_targets))

    logger.info(f'AI hole detection found {len(holes)} holes')
    success = True
    if len(holes) < success_threshold:
        success = False
    logger.debug(f'{holes[0]},{type(holes[0])}')
    
    holes = [(np.array(hole[0:-1])-np.array(list(montage.center)*2))*binning + np.array(list(montage.center)*2) for hole in holes]
    # logger.debug(f'{holes[0]},{type(holes[0])}')
    return holes, success, dict()


def find_and_classify_holes(montage, **kwargs):
    logger.info('Running AI hole detection and classification')
    # centroid = find_square_center(montage.image)
    holes, labels = detect_and_classify_holes(montage.image, **kwargs)
    # print(holes)
    success = True
    if len(holes) < 20:
        success = False

    logger.info(f'AI hole detection found {len(holes)} holes')
    return (holes, labels), success, 'AIHoleTarget'
