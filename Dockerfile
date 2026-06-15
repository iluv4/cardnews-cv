# Fully reproducible CV training env for cloud GPU (Linux).
# Build:  docker build -t cardnews-cv .
# Run  :  docker run --gpus all -it -v ${PWD}:/work cardnews-cv
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /work
# system libs for opencv / easyocr
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt .
# torch/torchvision already in the base image; install the rest.
RUN pip install --no-cache-dir -r requirements-lock.txt || true

COPY . .
CMD ["bash"]
