FROM tensorflow/tensorflow:latest

RUN apt update && \
    apt install -y --no-install-recommends \
      libgl1-mesa-dev \
      x11-apps

RUN pip3 install \
    opencv-python

WORKDIR /src
