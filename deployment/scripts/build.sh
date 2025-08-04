#!/bin/bash

# Build and run script for California Housing Model
# This script builds the Docker image and starts the services using docker-compose

set -e

echo "🔨 Building California Housing Model..."

# Change to the deployment directory
cd "$(dirname "$0")/.."

# Stop and remove existing containers
echo "🛑 Stopping and removing existing containers..."
docker-compose down || true

# Remove any existing images to ensure fresh build
echo "🧹 Cleaning up existing images..."
docker rmi california-housing:latest 2>/dev/null || true

# Build the new images and run the containers
echo "🚀 Building and running the new containers..."
docker-compose up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 30

# Test the application
echo "🧪 Testing the application..."
curl -f http://localhost:5001/health || echo "Health check failed"

echo "✅ Build and deployment completed!"
echo "📊 Access the application:"
echo "   API: http://localhost:5001"
echo "   Health: http://localhost:5001/health"
echo "   Prometheus: http://localhost:9090"
echo "   Grafana: http://localhost:3000 (admin/admin)"
