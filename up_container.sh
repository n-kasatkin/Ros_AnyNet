docker build .
docker run -it --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --privileged \
    -v $(pwd)/src:/usr/src/ \
    anynet_ros_anynet bash