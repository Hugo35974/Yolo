
FROM ubuntu:20.04

RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg
RUN apt upgrade --no-install-recommends -y openssl
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    mesa-utils \
    libglib2.0-0

RUN git clone https://github.com/Hugo35974/Yolo.git


RUN pip install gitpython>=3.1.30 
RUN pip install matplotlib>=3.3
RUN pip install numpy>=1.23.5
RUN pip install opencv-python>=4.1.1
RUN pip install Pillow>=9.4.0
RUN pip install psutil  # system resources
RUN pip install PyYAML>=5.3.1
RUN pip install requests>=2.23.0
RUN pip install scipy>=1.4.1
RUN pip install thop>=0.1.1  # FLOPs computation
RUN pip install torch>=1.8.0  # see https://pytorch.org/get-started/locally (recommended)
RUN pip install torchvision>=0.9.0
RUN pip install tqdm>=4.64.0
RUN pip install ultralytics>=8.0.232


# Exécutez votre application au démarrage
CMD ["python3 Yolo/main.py"]