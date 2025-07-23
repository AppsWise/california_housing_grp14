#!/bin/bash
CONTAINER_NAME="mlops-assignment"

# Stop and remove the existing container if it exists
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Stopping and removing existing container..."
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

# Build the new image
echo "Building the docker image..."
docker build -t mlops-assignment .

# Run the new container
echo "Running the new container..."
docker run -d -p 5000:5000 --name ${CONTAINER_NAME} mlops-assignment
