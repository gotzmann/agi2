# FROM cr.msk.sbercloud.ru/aicloud-base-images-test/cuda11.7-torch2:fdf9bece-630252

# MLSPACE_IMAGE_PARENT=nvidia/cuda:-devel-ubuntu20.04
# MLSPACE_IMAGE_NAME=cuda11.7-torch2
FROM cr.msk.sbercloud.ru/aijcontest_official/fbc3_0:0.1 as base
USER root
WORKDIR /app

# Ubuntu 20.04.6 LTS
# Python 3.9.16

# -- Build, tag, push and run image
# sudo docker build --tag supermachina:0.5 .
# sudo docker tag supermachina:0.5 cr.msk.sbercloud.ru/aijcontest/supermachina:0.5
# sudo docker push cr.msk.sbercloud.ru/aijcontest/supermachina:0.5
# sudo docker run --rm -it supermachina:0.5 -- sh

# -- Build for multi platforms
# sudo docker buildx build --platform linux/amd64 -f ./Dockerfile --tag supermachina:0.2 .

# -- Show and prune Docker cache
# sudo docker system df
# sudo docker builder prune

# -- Show and remove unused images
# sudo docker image ls
# sudo docker image rm supermachina:0.1

# -- Show TOP 20 biggest files and folders
# sudo du -ah / | sort -rh | head -n 20

# -- Show which process occupied some local port
# sudo lsof -i:8888 -P -n | grep LISTEN

# -- Reset GPU
# nvidia-smi --gpu-reset

# -- Show and kill processes using GPU
# lsof | grep /dev/nvidia

COPY model.gguf /app/model.gguf
COPY projection_LLaMa-7b-EN-Linear-ImageBind /app/projection_LLaMa-7b-EN-Linear-ImageBind
# COPY imagebind_huge.pth /app/imagebind_huge.pth
# COPY imagebind_huge.pth /app/.checkpoints/imagebind_huge.pth

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y --no-install-recommends mc nano git htop lsof make build-essential python3-pip

RUN wget https://golang.org/dl/go1.20.linux-amd64.tar.gz && \
    tar -xf go1.20.linux-amd64.tar.gz -C /usr/local

# RUN git clone https://github.com/gotzmann/llamazoo.git && \
#     cd ./llamazoo && \
#     LLAMA_CUBLAS=1 PATH=$PATH:/usr/local/go/bin CUDA_PATH=/usr/local/cuda CUDA_DOCKER_ARCH=sm_80 make -j cuda && \
#     mkdir /app && \
#     cp llamazoo /app/llamazoo && \
#     chmod +x /app/llamazoo

RUN mkdir -p /app/git && \
    cd /app/git && \
    git clone https://github.com/gotzmann/llamazoo.git && \
    cd ./llamazoo && \
    PATH=$PATH:/usr/local/go/bin make -j llamazoo && \
    cp llamazoo /app/llamazoo && \
    chmod +x /app/llamazoo

# json, time, traceback : standard python lib
# numpy : Requirement already satisfied: numpy in /home/user/conda/lib/python3.9/site-packages (from -r requirements.txt (line 3)) (1.24.1)

#RUN pip install requests
#RUN pip install sentencepiece
#RUN pip install https://github.com/enthought/mayavi/zipball/master
#RUN pip install --upgrade git+https://github.com/lizagonch/ImageBind.git aac_datasets torchinfo

# RUN pip install --no-cache-dir -r /app/requirements.txt

#COPY ./Llama-2-7B-fp16 ./Llama-2-7B-fp16
# COPY --from=base /app/model.gguf /app/model.gguf
# COPY --from=base /app/imagebind_huge.pth /app/imagebind_huge.pth
# COPY --from=base /app/projection_LLaMa-7b-EN-Linear-ImageBind /app/projection_LLaMa-7b-EN-Linear-ImageBind

RUN /bin/bash -cu pip install requests # buildkit
RUN /bin/bash -cu pip install sentencepiece # buildkit
RUN /bin/bash -cu pip install https://github.com/enthought/mayavi/zipball/master # buildkit
RUN /bin/bash -cu pip install --upgrade git+https://github.com/lizagonch/ImageBind.git aac_datasets torchinfo # buildkit
RUN /bin/bash -cu pip install protobuf==3.20.0 # buildkit
RUN /bin/bash -cu pip install -U transformers # buildkit
RUN /bin/bash -cu pip install peft==0.4.0 # buildkit

# -- See standard Python libs: https://docs.python.org/3/library/index.html
COPY requirements.txt /app/requirements.txt
RUN /bin/bash -cu pip install --no-cache-dir -r /app/requirements.txt # buildkit

COPY config.yaml /app/config.yaml

USER jovyan
WORKDIR /home/jovyan
