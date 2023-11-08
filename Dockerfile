# Ubuntu 20.04.6 LTS
# Python 3.9.16

FROM cr.msk.sbercloud.ru/aicloud-base-images-test/cuda11.7-torch2:fdf9bece-630252

USER root

# RUN apt update -y && \
#    apt upgrade -y && \
#    apt install -y mc nano htop lsof make build-essential

RUN apt install -y mc

RUN wget https://golang.org/dl/go1.20.linux-amd64.tar.gz && \
    tar -xf go1.20.linux-amd64.tar.gz -C /usr/local

# PATH=$PATH:/usr/local/go/bin LLAMA_CUBLAS=1 PATH=$PATH:/usr/local/go/bin CUDA_PATH=/usr/local/cuda-11 CUDA_DOCKER_ARCH=sm_80 make cuda && \

RUN git clone https://github.com/gotzmann/llamazoo.git && \
    cd ./llamazoo && \
    PATH=$PATH:/usr/local/go/bin make llamazoo && \
    mkdir /app && \
    cp llamazoo /app/llamazoo && \
    chmod +x /app/llamazoo

# RUN git clone https://github.com/gotzmann/agi.git

WORKDIR /app

#COPY imagebind_huge.pth .checkpoints/imagebind_huge.pth

# json, time, traceback : standard python lib
# numpy : Requirement already satisfied: numpy in /home/user/conda/lib/python3.9/site-packages (from -r requirements.txt (line 3)) (1.24.1)

COPY requirements.txt ./

RUN pip install https://github.com/enthought/mayavi/zipball/master
RUN pip install --upgrade git+https://github.com/lizagonch/ImageBind.git aac_datasets torchinfo
RUN pip install --no-cache-dir -r requirements.txt

#COPY ./Llama-2-7B-fp16 ./Llama-2-7B-fp16

#USER jovyan
#WORKDIR /home/jovyan
