# MuseTalk - RunPod Serverless
# Base: pytorch oficial do Docker Hub (CUDA 12.1)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/workspace/models \
    OUTPUT_DIR=/workspace/output \
    PYTHONPATH=/workspace

WORKDIR /workspace

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar projeto
COPY . /workspace

# Instalar dependências Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    runpod==1.6.2 \
    huggingface_hub==0.30.2 \
    diffusers==0.30.2 \
    accelerate==0.28.0 \
    numpy==1.23.5 \
    opencv-python==4.9.0.80 \
    soundfile==0.12.1 \
    transformers==4.39.2 \
    librosa==0.11.0 \
    einops==0.8.1 \
    omegaconf \
    ffmpeg-python \
    moviepy \
    gdown \
    requests \
    imageio[ffmpeg] \
    python-magic \
    mmpose \
    mmcv \
    mmdet

# Baixar pesos na build (evita cold start lento)
RUN mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper && \
    huggingface-cli download TMElyralab/MuseTalk \
      --local-dir models \
      --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin" && \
    huggingface-cli download TMElyralab/MuseTalk \
      --local-dir models \
      --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth" && \
    huggingface-cli download stabilityai/sd-vae-ft-mse \
      --local-dir models/sd-vae \
      --include "config.json" "diffusion_pytorch_model.bin" && \
    huggingface-cli download openai/whisper-tiny \
      --local-dir models/whisper \
      --include "config.json" "pytorch_model.bin" "preprocessor_config.json" && \
    huggingface-cli download yzd-v/DWPose \
      --local-dir models/dwpose \
      --include "dw-ll_ucoco_384.pth" && \
    huggingface-cli download ByteDance/LatentSync \
      --local-dir models/syncnet \
      --include "latentsync_syncnet.pt" && \
    gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth && \
    curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
      -o models/face-parse-bisent/resnet18-5c106cde.pth

CMD ["python", "-u", "handler.py"]
