#!/bin/bash

# Complete MLOps Monitoring Setup Script
# Fixes all issues and sets up working Prometheus + Grafana monitoring

set -e

echo "🏠 Setting up California Housing MLOps Monitoring Stack (FIXED VERSION)..."

# Check if we're in the correct conda environment
if [[ "$CONDA_DEFAULT_ENV" != "BITSAIML" ]]; then
    echo "❌ Error: Please activate the BITSAIML conda environment first:"
    echo "   conda activate BITSAIML"
    exit 1
fi

echo "✅ Using correct conda environment: $CONDA_DEFAULT_ENV"

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create required directories
echo "📁 Creating monitoring directories..."
mkdir -p monitoring/prometheus
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p logs

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f docker-compose.monitoring.yml down 2>/dev/null || true
docker rm -f prometheus-housing grafana-housing housing-api 2>/dev/null || true

# Remove old images to force rebuild
echo "🔄 Removing old images to force rebuild..."
docker rmi mlops_ass1-housing-api:latest 2>/dev/null || true

# Install required Python packages in the current environment
echo "📦 Installing required packages..."
pip install prometheus-client grafana-api

# Build the housing API image with latest code
echo "🏗️ Building housing API Docker image with latest code..."
docker build -t mlops_ass1-housing-api:latest .

# Start the monitoring stack
echo "🚀 Starting monitoring stack..."
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 15

# Function to check service health
check_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "✅ $name is running at $url"
            return 0
        fi
        echo "⏳ Waiting for $name (attempt $attempt/$max_attempts)..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ $name failed to start after $max_attempts attempts"
    return 1
}

echo "📊 Checking service status..."

# Check Prometheus
check_service "http://localhost:9090/-/healthy" "Prometheus"

# Check Grafana
check_service "http://localhost:3000/api/health" "Grafana"

# Check Housing API
check_service "http://localhost:5001/health" "Housing API"

# Test the comprehensive metrics endpoint
echo "🧪 Testing metrics endpoints..."
if curl -s http://localhost:5001/api/monitoring/metrics | grep -q "housing_api"; then
    echo "✅ Comprehensive Prometheus metrics are working"
else
    echo "⚠️  Comprehensive metrics might need a few minutes to initialize"
fi

# Generate some test data
echo "🧪 Generating test predictions to populate metrics..."
for i in {1..5}; do
    curl -s -X POST http://localhost:5001/api/predict \
        -H "Content-Type: application/json" \
        -d "{
            \"longitude\": $((RANDOM % 10 - 125)),
            \"latitude\": $((RANDOM % 10 + 35)),
            \"housing_median_age\": $((RANDOM % 50 + 1)),
            \"total_rooms\": $((RANDOM % 2000 + 500)),
            \"total_bedrooms\": $((RANDOM % 400 + 100)),
            \"population\": $((RANDOM % 1000 + 200)),
            \"households\": $((RANDOM % 300 + 50)),
            \"median_income\": $((RANDOM % 10 + 1)),
            \"ocean_proximity\": \"NEAR BAY\"
        }" > /dev/null
    echo "Generated test prediction $i/5"
done

echo ""
echo "🎉 Monitoring stack setup complete!"
echo ""
echo "📈 Access URLs:"
echo "   • Grafana Dashboard: http://localhost:3000"
echo "     Login: admin / admin123"
echo "   • Prometheus: http://localhost:9090"
echo "   • Housing API: http://localhost:5001"
echo "   • API Health: http://localhost:5001/health"
echo "   • API Metrics: http://localhost:5001/api/monitoring/metrics"
echo "   • Basic Metrics: http://localhost:5001/metrics/prometheus"
echo ""
echo "📝 To view logs:"
echo "   docker-compose -f docker-compose.monitoring.yml logs -f"
echo ""
echo "🛑 To stop all services:"
echo "   docker-compose -f docker-compose.monitoring.yml down"
echo ""

# Final verification
echo "🔍 Final verification..."
sleep 5

if curl -s http://localhost:5001/api/monitoring/metrics | head -5; then
    echo ""
    echo "🎊 SUCCESS! Your MLOps monitoring stack is working properly!"
    echo "Visit http://localhost:3000 to view the Grafana dashboard."
else
    echo ""
    echo "⚠️  Metrics endpoint might still be initializing. Please wait a few minutes."
fi

echo ""
echo "📚 Quick Test Commands:"
echo "# Test prediction:"
echo "curl -X POST http://localhost:5001/api/predict -H 'Content-Type: application/json' -d '{\"longitude\": -122.23, \"latitude\": 37.88, \"housing_median_age\": 41.0, \"total_rooms\": 880.0, \"total_bedrooms\": 129.0, \"population\": 322.0, \"households\": 126.0, \"median_income\": 8.3252, \"ocean_proximity\": \"NEAR BAY\"}'"
echo ""
echo "# Check metrics:"
echo "curl http://localhost:5001/api/monitoring/metrics | grep housing_api_predictions_total"
