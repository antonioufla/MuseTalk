# MuseTalk - RunPod Serverless
# Base: pytorch oficial do Docker Hub (CUDA 12.1)
# Updated: 2026-03-04 - Fix mmcv: mmdet requer mmcv>=2.0.0rc4,<2.2.0 → usar 2.1.0
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

# Instalar dependências Python (evitar conflitos na build)
RUN pip install --upgrade pip setuptools wheel && \
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
    python-magic

# Instalar mmengine, mmcv, mmdet, mmpose via openmim (método oficial OpenMMLab)
# chumpy (dep do mmpose) tem setup.py antigo com "import pip"; instalar sem isolamento antes
RUN pip install --no-cache-dir openmim && \
    pip install --no-build-isolation chumpy && \
    mim install "mmengine>=0.10.1" && \
    mim install "mmcv==2.1.0" && \
    mim install "mmdet==3.3.0" && \
    mim install "mmpose==1.3.2"

# Baixar pesos na build (evita cold start lento)
RUN mkdir -p models/musetalk models/musetalkV15 models/syncnet models/dwpose models/face-parse-bisent models/sd-vae models/whisper && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TMElyralab/MuseTalk', local_dir='models', allow_patterns=['musetalk/musetalk.json','musetalk/pytorch_model.bin'])" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TMElyralab/MuseTalk', local_dir='models', allow_patterns=['musetalkV15/musetalk.json','musetalkV15/unet.pth'])" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='stabilityai/sd-vae-ft-mse', local_dir='models/sd-vae', allow_patterns=['config.json','diffusion_pytorch_model.bin'])" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai/whisper-tiny', local_dir='models/whisper', allow_patterns=['config.json','pytorch_model.bin','preprocessor_config.json'])" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='yzd-v/DWPose', local_dir='models/dwpose', allow_patterns=['dw-ll_ucoco_384.pth'])" && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ByteDance/LatentSync', local_dir='models/syncnet', allow_patterns=['latentsync_syncnet.pt'])" && \
    gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth && \
    curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
      -o models/face-parse-bisent/resnet18-5c106cde.pth

CMD ["python", "-u", "handler.py"]
