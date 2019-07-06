from __future__ import absolute_import

from torchvision.transforms import *

#from PIL import Image
import random
import math
#import numpy as np
#import torch

def erasing(img, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
    for attempt in range(100):
        area = img.size()[1] * img.size()[2]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1/r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.size()[2] and h < img.size()[1]:
            x1 = random.randint(0, img.size()[1] - h)
            y1 = random.randint(0, img.size()[2] - w)
            if img.size()[0] == 3:
                img[0, x1:x1+h, y1:y1+w] = mean[0]
                img[1, x1:x1+h, y1:y1+w] = mean[1]
                img[2, x1:x1+h, y1:y1+w] = mean[2]
            else:
                img[0, x1:x1+h, y1:y1+w] = mean[0]
            return img
    return img

class RandomErasingPlus(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for i in range(5):
            img = erasing(img)
            if random.uniform(0, 1) > self.probability:
                return img

        return img

