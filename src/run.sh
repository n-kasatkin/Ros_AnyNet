# ROS PARAMETERS
LEFT_IMAGES="left_images_topic"
RIGHT_IMAGES="right_images_topic"
OUPUT_TOPIC="anynet_disparities"

# MODEL PARAMETERS
# SEE CODE FOR DETAILS
CHECKPOINT="/home/kasatkin/Projects/AnyNet/checkpoint/kitti2012_ck/checkpoint.tar"
DATATYPE=2012

python3 run.py ${LEFT_IMAGES} ${RIGHT_IMAGES} \
    --output_topic ${OUPUT_TOPIC} \
    --pretrained ${CHECKPOINT} \
    --datatype ${DATATYPE}