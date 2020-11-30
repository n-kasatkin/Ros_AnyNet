import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# CROP_H = 368
# CROP_W = 1232
CROP_H = 368
CROP_W = 1232

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, loader=default_loader):
 
        self.left = left
        self.right = right
        self.loader = loader

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]

        left_img = self.loader(left)
        right_img = self.loader(right)

        w, h = left_img.size

        # left_img = left_img.crop((w-CROP_W, h-CROP_H, w, h))
        # right_img = right_img.crop((w-CROP_W, h-CROP_H, w, h))
        # w1, h1 = left_img.size
        left_img = left_img.resize((CROP_W, CROP_H))
        right_img = right_img.resize((CROP_W, CROP_H))

        processed = preprocess.get_transform(augment=False)  
        left_img = processed(left_img)
        right_img = processed(right_img)

        return left_img, right_img

    def __len__(self):
        return len(self.left)
