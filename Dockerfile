# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install required dependencies
RUN pip install --no-cache-dir torch torchvision timm tqdm pillow

# Command to run the script (script and dataset will be mounted as a volume)
CMD ["python", "/app/model-2.py"]
