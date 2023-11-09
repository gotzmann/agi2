# Ubuntu 20.04.6 LTS
# Python 3.9.16

# FROM cr.msk.sbercloud.ru/aicloud-base-images-test/cuda11.7-torch2:fdf9bece-630252

# MLSPACE_IMAGE_PARENT=nvidia/cuda:-devel-ubuntu20.04
# MLSPACE_IMAGE_NAME=cuda11.7-torch2
FROM cr.msk.sbercloud.ru/aijcontest_official/fbc3_0:0.1 as base

COPY model.gguf /app/model.gguf
COPY imagebind_huge.pth /app/imagebind_huge.pth
COPY projection_LLaMa-7b-EN-Linear-ImageBind /app/projection_LLaMa-7b-EN-Linear-ImageBind

# RUN apt update -y && \
#     apt upgrade -y && \
#     apt install -y mc nano git htop lsof make build-essential

# RUN wget https://golang.org/dl/go1.20.linux-amd64.tar.gz && \
#     tar -xf go1.20.linux-amd64.tar.gz -C /usr/local

# RUN git clone https://github.com/gotzmann/llamazoo.git && \
#     cd ./llamazoo && \
#     LLAMA_CUBLAS=1 PATH=$PATH:/usr/local/go/bin CUDA_PATH=/usr/local/cuda CUDA_DOCKER_ARCH=sm_80 make -j cuda && \
#     mkdir /app && \
#     cp llamazoo /app/llamazoo && \
#     chmod +x /app/llamazoo

# json, time, traceback : standard python lib
# numpy : Requirement already satisfied: numpy in /home/user/conda/lib/python3.9/site-packages (from -r requirements.txt (line 3)) (1.24.1)

# COPY requirements.txt ./

# RUN pip install https://github.com/enthought/mayavi/zipball/master
# RUN pip install --upgrade git+https://github.com/lizagonch/ImageBind.git aac_datasets torchinfo
# RUN pip install --no-cache-dir -r requirements.txt

FROM base
USER root
WORKDIR /app

#COPY ./Llama-2-7B-fp16 ./Llama-2-7B-fp16

# COPY --from=base /app/model.gguf /app/model.gguf
# COPY --from=base /app/imagebind_huge.pth /app/imagebind_huge.pth
# COPY --from=base /app/projection_LLaMa-7b-EN-Linear-ImageBind /app/projection_LLaMa-7b-EN-Linear-ImageBind

COPY config.yaml        /app/config.yaml
COPY llamazoo           /app/llamazoo
RUN chmod +x            /app/llamazoo

# DEBUG
# ENTRYPOINT [ "./llamazoo", "--server", "--debug" ]

#USER jovyan
#WORKDIR /home/jovyan
