# MuJoCo environment based on official Python image
FROM python:3.12-slim-bookworm

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglew-dev \
    libegl1 \
    libgles2 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    xvfb \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY req.txt /tmp/req.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/req.txt

# Setup environment variables for MuJoCo
ENV MUJOCO_GL=osmesa

# Create mount point for workspace
RUN mkdir -p /container_workspace
WORKDIR /container_workspace

# Default command
CMD ["python", "train.py"]