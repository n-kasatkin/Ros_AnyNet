import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import cv2
import numpy as np
from . import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# Husky
CROP_H = 576
CROP_W = 1200

# # KITTI
# CROP_H = 368
# CROP_W = 1232


def analyze(img):
    print(f'Shape  | {img.shape}')
    print(f'Max    | {np.amax(img)}')
    print(f'Min    | {np.amin(img)}')
    print(f'Mean   | {np.mean(img.reshape(-1))}')
    print(f'Median | {np.median(img.reshape(-1))}')

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return cv2.imread(path[:-4] + '.exr', cv2.IMREAD_UNCHANGED)


class myImageFloder_testing(data.Dataset):
    def __init__(self, left, right, loader=default_loader):
        self.left = left
        self.right = right
        self.loader = loader

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]

        left_img = self.loader(left)
        right_img = self.loader(right)

        left_img = left_img.resize((CROP_W, CROP_H))
        right_img = right_img.resize((CROP_W, CROP_H))

        processed = preprocess.get_transform(augment=False)  
        left_img = processed(left_img)
        right_img = processed(right_img)

        return left_img, right_img

    def __len__(self):
        return len(self.left)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)


        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = dataL[y1:y1 + th, x1:x1 + tw]

           processed = preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)
           return left_img, right_img, dataL
        else:
           w, h = left_img.size

           left_img = left_img.resize((CROP_W, CROP_H))
           right_img = right_img.resize((CROP_W, CROP_H))
           w1, h1 = left_img.size

           dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
           dataL = cv2.resize(dataL, (CROP_W, CROP_H))

           processed = preprocess.get_transform(augment=False)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)
           return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
