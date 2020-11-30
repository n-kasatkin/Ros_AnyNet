#!/usr/bin/python3
import rospy
import ros_numpy
import os
import torch
import time
import json 
import yaml
import struct
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.point_cloud2 import create_cloud
import numpy as np
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

from torchsparse.utils import sparse_quantize, sparse_collate
from torchsparse import SparseTensor
from model_zoo import minkunet, spvcnn, spvnas_specialized

def create_label_map(num_classes=19):
    name_label_mapping = {
        'unlabeled': 0, 'outlier': 1, 'car': 10, 'bicycle': 11,
        'bus': 13, 'motorcycle': 15, 'on-rails': 16, 'truck': 18,
        'other-vehicle': 20, 'person': 30, 'bicyclist': 31,
        'motorcyclist': 32, 'road': 40, 'parking': 44,
        'sidewalk': 48, 'other-ground': 49, 'building': 50,
        'fence': 51, 'other-structure': 52, 'lane-marking': 60,
        'vegetation': 70, 'trunk': 71, 'terrain': 72, 'pole': 80,
        'traffic-sign': 81, 'other-object': 99, 'moving-car': 252,
        'moving-bicyclist': 253, 'moving-person': 254, 'moving-motorcyclist': 255,
        'moving-on-rails': 256, 'moving-bus': 257, 'moving-truck': 258,
        'moving-other-vehicle': 259
    }
    
    for k in name_label_mapping:
        name_label_mapping[k] = name_label_mapping[k.replace('moving-', '')]
    train_label_name_mapping = {
        0: 'car', 1: 'bicycle', 2: 'motorcycle', 3: 'truck', 4:
        'other-vehicle', 5: 'person', 6: 'bicyclist', 7: 'motorcyclist',
        8: 'road', 9: 'parking', 10: 'sidewalk', 11: 'other-ground',
        12: 'building', 13: 'fence', 14: 'vegetation', 15: 'trunk',
        16: 'terrain', 17: 'pole', 18: 'traffic-sign'
    }

    label_map = np.zeros(260)+num_classes
    for i in range(num_classes):
        cls_name = train_label_name_mapping[i]
        label_map[name_label_mapping[cls_name]] = min(num_classes,i)
    return label_map.astype(np.int64)

# models -- just in case!
NOT_USED_model_list = [        
    "SemanticKITTI_val_MinkUNet@29GMACs",
    "SemanticKITTI_val_SPVCNN@30GMACs",
    "SemanticKITTI_val_SPVNAS@20GMACs",
    "SemanticKITTI_val_SPVNAS@25GMACs",
    "SemanticKITTI_val_MinkUNet@46GMACs",
    "SemanticKITTI_val_SPVCNN@47GMACs",
    "SemanticKITTI_val_SPVNAS@35GMACs",
    "SemanticKITTI_val_MinkUNet@114GMACs",
    "SemanticKITTI_val_SPVCNN@119GMACs",
    "SemanticKITTI_val_SPVNAS@65GMACs"
]

class Processor_ROS:
    def __init__(self):
        # Model Loading
        rospy.loginfo("Model Loading!")
        model_name = rospy.get_param("~model_name")
        rospy.logdebug(f"Model {model_name} is loaded!!")
        global model 
        if 'MinkUNet' in model_name:
            model = minkunet(model_name, pretrained=True)
        elif 'SPVCNN' in model_name:
            model = spvcnn(model_name, pretrained=True)
        elif 'SPVNAS' in model_name:
            model = spvnas_specialized(model_name, pretrained=True)
        else:
            raise NotImplementedError
        model = model.to(device)
        
    def initialize(self):
        rospy.loginfo("Deploy OK!")

def process_point_cloud(msg, input_labels=None, voxel_size=0.05):
    t_t = time.time()
    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    point_cloud = get_xyz_points(msg_cloud, True)
    
    #input_point_cloud = point_cloud.reshape([-1,5])
    input_point_cloud = point_cloud[:,:4]    
    input_point_cloud[:, 3] = input_point_cloud[:, 3]
    pc_ = np.round(input_point_cloud[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)
    
    label_map = create_label_map()
    if input_labels is not None:
        labels_ = label_map[input_labels & 0xFFFF].astype(
            np.int64)  # semantic labels
    else:
        labels_ = np.zeros(pc_.shape[0], dtype=np.int64)
    
    feat_ = input_point_cloud
    
    if input_labels is not None:
        out_pc = input_point_cloud[labels_ != labels_.max(), :3]
        pc_ = pc_[labels_ != labels_.max()]
        feat_ = feat_[labels_ != labels_.max()]
        labels_ = labels_[labels_ != labels_.max()]
    else:
        out_pc = input_point_cloud
        pc_ = pc_
        
    inds, labels, inverse_map = sparse_quantize(pc_,
                                                feat_,
                                                labels_,
                                                return_index=True,
                                                return_invs=True)
    pc = np.zeros((inds.shape[0], 4))
    pc[:, :3] = pc_[inds]
    
    feat = feat_[inds]
    labels = labels_[inds]
    lidar = SparseTensor(
        torch.from_numpy(feat).float(), 
        torch.from_numpy(pc).int()
    )
    feed_dict = {
        'pc': out_pc,
        'lidar': lidar,
        'targets': labels,
        'targets_mapped': labels_,
        'inverse_map': inverse_map
    }

    inputs = feed_dict['lidar'].to(device)
    t = time.time()
    outputs = model(inputs)
    rospy.logdebug("Network predict time cost:", time.time() - t)
    predictions = outputs.argmax(1).cpu().numpy()
    predictions = predictions[feed_dict['inverse_map']]    # Here you can check segmentaion results for each point
    #rospy.loginfo(predictions)
    input_point_cloud =input_point_cloud.astype(np.float32)
    #sending segmented pc
    pc_id = predictions.reshape(-1)
    labels = pc_id.astype(np.uint32)
    msg = to_msg(input_point_cloud[:,:3], labels, msg.header)
    pub_.publish(msg)

    rospy.logdebug(f"Total cost time: {time.time() - t_t}")
    return 

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    return points		 

def spvnas2kitti(label):
    kitti_class = spvnas_labels[label]
    kitti_label = kitti_classes[kitti_class]
    return kitti_label

def to_rgb(label):
    label = spvnas2kitti(label)

    if label not in cfg['color_map']:
        rospy.logdebug('Error label: %s' % label)
        label = 0
    color = cfg['color_map'][label]
    rgb = struct.unpack('I', struct.pack('BBBB', color[2], color[1], color[0], 255))[0]
    return rgb

def set_color(point, label):
    return point.tolist() + [label]

def to_msg(points, labels, header):
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
    ]
    # getting color for each point by its label
    rgb_labels = list(map(lambda label: to_rgb(label), labels))
    # changing label of each point to its color
    color_points = list(map(lambda point, rgb_label: set_color(point, rgb_label), points, rgb_labels))    
    msg = create_cloud(header, fields, color_points)
    return msg

cfg = None
kitti_classes = json.load(open('/home/docker_spvnas/catkin_ws/src/spvnas/spvnas_ros_node/src/kitti_classes.json'))
spvnas_labels = { int(k) : v for k, v in json.load(open('/home/docker_spvnas/catkin_ws/src/spvnas/spvnas_ros_node/src/spvnas_labels.json')).items() }
cfg = yaml.safe_load(open('/home/docker_spvnas/catkin_ws/src/spvnas/spvnas_ros_node/src/semantic-kitti.yaml', 'r'))

if __name__ == "__main__":    
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    ## SPVNAS
    rospy.init_node('SPVNAS_ros_node')
    proc = Processor_ROS()
    proc.initialize()
    sub_ = rospy.Subscriber("pointcloud", PointCloud2, process_point_cloud, queue_size=1, buff_size=2**24)
    pub_ = rospy.Publisher("lidar_pc", PointCloud2, queue_size=1)
    rospy.loginfo("[+] SPVNAS 3D semantic segmentation ROS-node has started!")   
    rospy.spin()

