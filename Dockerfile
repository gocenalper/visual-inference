# Use NVIDIA's CUDA base image with PyTorch support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install required dependencies, ensuring PyTorch is GPU-enabled
RUN pip install --no-cache-dir torch torchvision timm tqdm pillow --extra-index-url https://download.pytorch.org/whl/cu121

# Command to run the script (script and dataset will be mounted as a volume)
CMD ["python3", "/app/model-2.py"]
