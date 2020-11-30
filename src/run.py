import os
import sys
sys.path.append('./AnyNet')
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import time
import json 
import yaml
import argparse
import numpy as np

import torch

import rospy
import ros_numpy

import struct
from std_msgs.msg import Header
from sensor_msgs.msg import Image

from AnyNet.main import AnyNetModel, add_model_specific_args


def add_ros_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('left_images', help='topic with left images')
    parser.add_argument('right_images', help='topic with right images')
    parser.add_argument('--output_topic', default='anynet_disparities',
                        help='topic to post disparities')
    return parser


def config_args():
    parser = argparse.ArgumentParser()

    parser = add_ros_specific_args(parser)
    parser = add_model_specific_args(parser)

    return parser.parse_args()


def sub_callback(data):
    rospy.loginfo("Got message %s", data.data)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "Cuda seems not to work"

    args = config_args()

    model = AnyNetModel(args)

    # Init AnyNet ros node
    rospy.init_node('AnyNet_ros_node')

    left_img_sub_ = rospy.Subscriber(args.left_images, Image, sub_callback, queue_size=1)
    right_img_sub_ = rospy.Subscriber(args.right_images, Image, sub_callback, queue_size=1)
    pub_ = rospy.Publisher("anynet_disparities", Image, queue_size=1)
    
    rospy.loginfo("[+] AnyNet ROS-node has started!")   
    # rospy.spin()