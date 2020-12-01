import os
import sys
sys.path.append('./AnyNet')

import time
import json 
import yaml
import argparse
import numpy as np
import cv2

import torch

import rospy
import ros_numpy
from cv_bridge import CvBridge

import struct
import message_filters
from std_msgs.msg import Header
from sensor_msgs.msg import Image

from AnyNet.main import AnyNetModel, add_model_specific_args
from AnyNet.dataloader import preprocess


def add_ros_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('left_images', help='topic with left images')
    parser.add_argument('right_images', help='topic with right images')
    parser.add_argument('--output_topic', default='anynet_disparities',
                        help='topic to post disparities')
    return parser


def add_data_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--input_w', type=int,
        help="Image shapes should be divisible by 16 due to model architecture")
    parser.add_argument('--input_h', type=int,
        help="Image shapes should be divisible by 16 due to model architecture")
    return parser


def config_args():
    parser = argparse.ArgumentParser()

    parser = add_ros_specific_args(parser)
    parser = add_data_specific_args(parser)
    parser = add_model_specific_args(parser)

    return parser.parse_args()


def preprocess_image(image):
    img = cv2.resize(image, (args.input_w, args.input_h))
    processed = preprocess.get_transform(augment=False)
    img = processed(img)
    return img


# See example http://wiki.ros.org/message_filters#Example_.28Python.29-1
def callback(imgL, imgR, verbose=False, quiet=False):
    if not quiet:
        rospy.loginfo(f'Got pair of images.\nLeft image info: {imgL.header}')

    if verbose:
        rospy.loginfo('input_msg height  : {}'.format(imgL.height))
        rospy.loginfo('input_msg width   : {}'.format(imgL.width))
        rospy.loginfo('input_msg step    : {}'.format(imgL.step))
        rospy.loginfo('input_msg encoding: {}'.format(imgL.encoding))

    imgL = br.imgmsg_to_cv2(imgL, desired_encoding='rgb8')
    imgR = br.imgmsg_to_cv2(imgR, desired_encoding='rgb8')

    # Preprocess
    imgL = preprocess_image(imgL)
    imgR = preprocess_image(imgR)

    # Predict disparity
    disparity = model.predict_disparity(imgL, imgR)
    disparity = disparity.cpu().numpy()[0, :, :]

    # Convert to msg and publish
    msg = br.cv2_to_imgmsg(disparity)
    pub_.publish(msg)

    # cv2.imwrite('output.png', disparity)

    if verbose:
        rospy.loginfo('-'*50)
        rospy.loginfo('output type       : {}'.format(type(disparity)))
        rospy.loginfo('output dtype      : {}'.format(disparity.dtype))
        rospy.loginfo('output shape      : {}'.format(disparity.shape))


if __name__ == "__main__":
    assert torch.cuda.is_available(), "Cuda seems not to work"

    args = config_args()

    model = AnyNetModel(args)

    # Init AnyNet ros node
    rospy.init_node('AnyNet_ros_node')

    br = CvBridge()

    left_img_sub_ = message_filters.Subscriber(args.left_images, Image)
    right_img_sub_ = message_filters.Subscriber(args.right_images, Image)
    pub_ = rospy.Publisher(args.output_topic, Image, queue_size=1)

    # Syncronization
    ts = message_filters.ApproximateTimeSynchronizer([left_img_sub_, right_img_sub_], 1, 1)
    ts.registerCallback(callback)

    rospy.loginfo("[+] AnyNet ROS-node has started!")   
    rospy.spin()