FROM nvidia/cuda:11.0-devel-ubuntu20.04

#install required libs and packages
RUN apt-get update  \
    && apt-get install -y --no-install-recommends build-essential git curl ca-certificates\
    libsparsehash-dev python3-dev libjpeg-dev libpng-dev python3-pip  \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA cu110

# CUDA 11.0 path
ENV PATH=/usr/local/cuda-11.0/bin:$PATH
ENV CUDA_PATH=/usr/local/cuda-11.0
ENV CUDA_HOME=/usr/local/cuda-11.0
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

# Nvidia driver
# RUN apt install nvidia-driver-450

# ROS Noetic installation
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y lsb-release && apt-get clean all
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt-get update  \
    && apt-get install -y ros-noetic-desktop
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash"
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"

# Pytorch installation
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install wheel
RUN pip install torch===1.7.0+cu110 torchvision===0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /root

# ROS_numpy
RUN git clone https://github.com/eric-wieser/ros_numpy.git \
    && cd ros_numpy  \
    && python3 setup.py install \
    && cd .. \ 
    && rm -r ros_numpy

WORKDIR /usr/src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt