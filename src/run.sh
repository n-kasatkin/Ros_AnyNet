# ROS PARAMETERS
LEFT_IMAGES="/left/image_raw"
RIGHT_IMAGES="/right/image_raw"
OUPUT_TOPIC="/anynet_disparities"

# DATA PARAMETERS
INPUT_W=1200
INPUT_H=576

# MODEL PARAMETERS
# SEE CODE FOR DETAILS
CHECKPOINT="./AnyNet/checkpoint/kitti2012_ck/checkpoint.tar"
DATATYPE=2012


python3 run.py ${LEFT_IMAGES} ${RIGHT_IMAGES} \
    --output_topic ${OUPUT_TOPIC} \
    --pretrained ${CHECKPOINT} \
    --datatype ${DATATYPE} \
    --input_w ${INPUT_W} \
    --input_h ${INPUT_H} \
    --with_spn