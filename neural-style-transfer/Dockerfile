FROM ubuntu:xenial

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libcurl4-openssl-dev python-pip libboost-python-dev build-essential cmake pkg-config && \
    apt-get install -y --no-install-recommends libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev && \
    apt-get install -y --no-install-recommends libxvidcore-dev libx264-dev libgtk2.0-dev libatlas-base-dev gfortran python2.7-dev && \
    apt-get install wget
#rm -rf /var/lib/apt/lists/* 

RUN pip install setuptools
#COPY requirements.txt ./
#RUN pip install -r requirements.txt

ENV OPENCV_VERSION=3.3.0

RUN apt install -y wget unzip

# Install OpenCV
RUN cd /opt && \
    wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    rm -rf ${OPENCV_VERSION}.zip && \
    mkdir -p /opt/opencv-${OPENCV_VERSION}/build && \
    cd /opt/opencv-${OPENCV_VERSION}/build && \
    cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_FFMPEG=NO \
    -D WITH_IPP=NO \
    -D WITH_OPENEXR=NO \
    -D WITH_TBB=YES \
    -D BUILD_EXAMPLES=NO \
    -D BUILD_ANDROID_EXAMPLES=NO \
    -D INSTALL_PYTHON_EXAMPLES=NO \
    -D BUILD_DOCS=NO \
    -D BUILD_opencv_python2=ON \
    -D BUILD_opencv_python3=NO \
    .. && \
    make VERBOSE=1 && \
    make -j${CPUCOUNT} && \
    make install && \
    rm -rf /opt/opencv-${OPENCV_VERSION} && \
    ln -s /usr/local/lib/python2.7/site-packages/cv2.cpython-36m-x86_64-linux-gnu.so /usr/local/lib/python2.7/site-packages/cv2.so

RUN pip install imutils numpy opencv-python

COPY . .

RUN useradd -ms /bin/bash moduleuser
USER moduleuser

CMD [ "python", "-u", "./neural_style_transfer_video.py", "-m", "models/eccv16"]
