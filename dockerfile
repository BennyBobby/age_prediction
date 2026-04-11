# 1. On passe à CUDA 12.8 (Crucial pour la série 50)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. On garde tes installations mais on simplifie un peu
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.12 /usr/bin/python

WORKDIR /app
RUN git config --global --add safe.directory /app

# Installation de uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. LA LIGNE MAGIQUE : Installer PyTorch compatible Blackwell dès le build
# On l'installe en mode "system" pour que uv sync ne le remplace pas par une version instable
RUN uv pip install --system --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# On copie le reste
COPY . .

# On synchronise le reste des dépendances (sans toucher à torch)
RUN uv sync