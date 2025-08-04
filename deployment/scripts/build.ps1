$containerName = "mlops-assignment"

# Check if a container with the same name exists
$existingContainer = docker ps -a -q --filter "name=$containerName"
if ($existingContainer) {
    Write-Host "Stopping and removing existing container..."
    docker stop $containerName
    docker rm $containerName
}

Write-Host "Building the Docker image..."
docker build -t mlops-assignment .

Write-Host "Running a new container..."
docker run -d -p 5000:5000 --name $containerName mlops-assignment
