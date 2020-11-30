import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import time
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import utils.logger as logger
from dataloader import loader_husky as DataLoader_
import models.anynet


def add_model_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
    parser.add_argument('--max_disparity', type=int, default=192)
    parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
    parser.add_argument('--datatype', default='2015',
                        help='datapath')
    parser.add_argument('--datapath', default=None, help='datapath')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--train_bsize', type=int, default=6,
                        help='batch size for training (default: 6)')
    parser.add_argument('--test_bsize', type=int, default=8,
                        help='batch size for testing (default: 8)')
    parser.add_argument('--save_path', type=str, default='results/finetune_anynet',
                        help='the path of saving checkpoints and log')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume path')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
    parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
    parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
    parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
    parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
    parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
    parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
    parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
    parser.add_argument('--start_epoch_for_spn', type=int, default=121)
    parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                        help='pretrained model path')
    parser.add_argument('--split_file', type=str, default=None)
    parser.add_argument('--evaluate', action='store_true')

    return parser


class AnyNetModel:
    def __init__(self, args):
        self.setup(args)
        self.stages = 3 + args.with_spn
        self.model.eval()

    def setup(self, args):
        self.model = models.anynet.AnyNet(args)
        self.model = nn.DataParallel(self.model).cuda()

        optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))

        if args.pretrained:
            if os.path.isfile(args.pretrained):
                checkpoint = torch.load(args.pretrained)
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                print("=> no pretrained model found at '{}'")
        cudnn.benchmark = True

    def predict_disparity(self, imgL, imgR, stage=4):
        imgL = imgL.float().cuda().unsqueeze(0)
        imgR = imgR.float().cuda().unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(imgL, imgR)
            output = torch.squeeze(outputs[stage - 1], 1)

        return output