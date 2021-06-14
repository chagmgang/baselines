FROM nvidia/cuda:10.0-base-ubuntu16.04
FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN pip install opencv-python
RUN apt-get update
RUN pip install gym
RUN apt-get install -y git
RUN pip install git+https://github.com/Kojoley/atari-py.git
RUN pip install tensorboardX
RUN apt-get update
RUN apt install -y libsm6 libxext6 libxrender-dev

COPY . /app

WORKDIR /app
