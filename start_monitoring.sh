#!/bin/bash

# MLOps Monitoring Setup Script
# Sets up and starts Prometheus, Grafana, and Housing API with monitoring

set -e

echo "🏠 Setting up California Housing MLOps Monitoring Stack..."

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

# Create .env file for environment variables
echo "🔧 Creating environment configuration..."
cat > .env << EOF
# MLOps Monitoring Environment Variables
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
API_PORT=5001
GRAFANA_ADMIN_PASSWORD=admin123
FLASK_ENV=production
PROMETHEUS_ENABLED=true
EOF

# Stop any existing services
echo "🛑 Stopping existing services..."
docker-compose -f docker-compose.monitoring.yml down 2>/dev/null || true

# Remove old containers if they exist
echo "🧹 Cleaning up old containers..."
docker rm -f prometheus-housing grafana-housing housing-api 2>/dev/null || true

# Build the housing API image
echo "🏗️ Building housing API Docker image..."
docker build -t housing-api:latest .

# Start the monitoring stack
echo "🚀 Starting monitoring stack..."
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "📊 Checking service status..."

# Check Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus is running at http://localhost:9090"
else
    echo "❌ Prometheus failed to start"
fi

# Check Grafana
if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana is running at http://localhost:3000"
    echo "   📝 Login: admin / admin123"
else
    echo "❌ Grafana failed to start"
fi

# Check Housing API
if curl -s http://localhost:5001/health > /dev/null; then
    echo "✅ Housing API is running at http://localhost:5001"
else
    echo "❌ Housing API failed to start"
fi

echo ""
echo "🎉 Monitoring stack setup complete!"
echo ""
echo "📈 Access URLs:"
echo "   • Grafana Dashboard: http://localhost:3000 (admin/admin123)"
echo "   • Prometheus: http://localhost:9090"
echo "   • Housing API: http://localhost:5001"
echo "   • API Metrics: http://localhost:5001/api/monitoring/metrics"
echo ""
echo "📝 To view logs:"
echo "   docker-compose -f docker-compose.monitoring.yml logs -f"
echo ""
echo "🛑 To stop all services:"
echo "   docker-compose -f docker-compose.monitoring.yml down"
echo ""

# Test the monitoring setup
echo "🧪 Testing monitoring setup..."

# Test a prediction to generate metrics
echo "Making a test prediction to generate metrics..."
curl -s -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }' > /dev/null && echo "✅ Test prediction successful" || echo "❌ Test prediction failed"

# Check if metrics are being generated
echo "Checking Prometheus metrics..."
if curl -s http://localhost:5001/api/monitoring/metrics | grep -q "housing_api"; then
    echo "✅ Prometheus metrics are being generated"
else
    echo "❌ Prometheus metrics not found"
fi

echo ""
echo "🎊 Setup complete! Your MLOps monitoring stack is ready."
echo "Visit http://localhost:3000 to view the Grafana dashboard."
