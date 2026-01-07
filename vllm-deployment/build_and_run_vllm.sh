#!/bin/bash

set -e

IMAGE_NAME="vllm-deployment"
CONTAINER_NAME="vllm-server"
DOCKERFILE="Dockerfile.vllm"

# Configuration
PORT="${PORT:-8000}"  # Port mapping
WORKSPACE_PATH="${WORKSPACE_PATH:-/raid/scratch/shaun_sit/vllm-deployment}"  # Mount point for files

echo "Building vLLM Docker image..."

# Build the Docker image from the Dockerfile
docker build -f ${DOCKERFILE} -t ${IMAGE_NAME} .

echo "Image built successfully!"

# Remove existing container if it exists
if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container: ${CONTAINER_NAME}"
    docker rm -f ${CONTAINER_NAME}
fi

# Create workspace directory if it doesn't exist
mkdir -p "${WORKSPACE_PATH}"

# Run the container
echo "Starting container..."
docker run --gpus all --name ${CONTAINER_NAME} -it \
    -p ${PORT}:8000 \
    -v ${WORKSPACE_PATH}:/workspace \
    -e LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH \
    ${IMAGE_NAME}

