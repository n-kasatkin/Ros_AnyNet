docker build .
docker run -it --gpus all -v $(pwd)/src:/usr/src/ anynet_ros_anynet bash