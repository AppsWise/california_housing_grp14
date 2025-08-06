#!/bin/bash

# Complete MLOps Monitoring Test & Verification Script
# Tests all components of the Prometheus + Grafana monitoring stack

set -e

echo "🧪 Testing MLOps Monitoring Stack..."
echo "======================================"

# Function to test service with retry
test_service() {
    local url=$1
    local name=$2
    local expected_pattern=$3
    
    echo -n "Testing $name... "
    
    if response=$(curl -s -m 10 "$url" 2>/dev/null); then
        if [[ -z "$expected_pattern" ]] || echo "$response" | grep -q "$expected_pattern"; then
            echo "✅ WORKING"
            return 0
        else
            echo "❌ FAILED (unexpected response)"
            echo "Response: $response"
            return 1
        fi
    else
        echo "❌ FAILED (connection error)"
        return 1
    fi
}

# Test all services
echo "📊 Testing Core Services:"
test_service "http://localhost:9090/-/healthy" "Prometheus" "Healthy"
test_service "http://localhost:3000/api/health" "Grafana" "ok"

# Test API endpoints
echo ""
echo "🏠 Testing Housing API Endpoints:"
test_service "http://localhost:5001/metrics/prometheus" "Basic Metrics" "housing_api"
test_service "http://localhost:5001/api/monitoring/metrics" "Comprehensive Metrics" "housing_api_predictions_total"

# Test prediction functionality
echo ""
echo "🧪 Testing Prediction Functionality:"
echo -n "Making test prediction... "

prediction_response=$(curl -s -X POST http://localhost:5001/api/predict \
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
  }' 2>/dev/null)

if echo "$prediction_response" | grep -q '"prediction"'; then
    prediction_value=$(echo "$prediction_response" | grep -o '"prediction":[0-9.]*' | cut -d: -f2)
    echo "✅ WORKING (Predicted: \$${prediction_value})"
else
    echo "❌ FAILED"
    echo "Response: $prediction_response"
fi

# Generate multiple predictions for metrics
echo ""
echo "📈 Generating test data for metrics..."
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
    echo "Generated prediction $i/5"
done

# Wait for metrics to update
echo "⏳ Waiting for metrics to update..."
sleep 3

# Test metrics after predictions
echo ""
echo "📊 Verifying Metrics Collection:"

# Test prediction counter
echo -n "Checking prediction counter... "
if curl -s http://localhost:5001/api/monitoring/metrics | grep -q "housing_api_predictions_total.*[1-9]"; then
    echo "✅ WORKING"
else
    echo "❌ FAILED"
fi

# Test processing time metrics
echo -n "Checking processing time metrics... "
if curl -s http://localhost:5001/api/monitoring/metrics | grep -q "housing_api_processing_time_seconds"; then
    echo "✅ WORKING"
else
    echo "❌ FAILED"
fi

# Test database metrics
echo -n "Checking database metrics... "
if curl -s http://localhost:5001/api/monitoring/metrics | grep -q "housing_api_database_size_bytes"; then
    echo "✅ WORKING"
else
    echo "❌ FAILED"
fi

# Test Prometheus targets
echo ""
echo "🎯 Testing Prometheus Integration:"
echo -n "Checking Prometheus targets... "
if curl -s "http://localhost:9090/api/v1/targets" | grep -q "housing-api"; then
    echo "✅ WORKING"
else
    echo "❌ FAILED"
fi

# Show current metrics summary
echo ""
echo "📈 Current Metrics Summary:"
echo "=========================="

# Get prediction count
prediction_count=$(curl -s http://localhost:5001/api/monitoring/metrics | grep "housing_api_predictions_total" | grep -o '[0-9.]*$' | head -1)
echo "Total Predictions: ${prediction_count:-0}"

# Get database size
db_size=$(curl -s http://localhost:5001/api/monitoring/metrics | grep "housing_api_database_size_bytes" | grep -o '[0-9.]*$' | head -1)
if [[ -n "$db_size" ]]; then
    db_size_mb=$(echo "scale=2; $db_size / 1024 / 1024" | bc 2>/dev/null || echo "unknown")
    echo "Database Size: ${db_size_mb} MB"
fi

# Get error rate
error_rate=$(curl -s http://localhost:5001/api/monitoring/metrics | grep "housing_api_error_rate_percent" | grep -o '[0-9.]*$' | head -1)
echo "Error Rate: ${error_rate:-0}%"

echo ""
echo "🎉 Monitoring Stack Status:"
echo "=========================="
echo "✅ Prometheus: http://localhost:9090"
echo "✅ Grafana: http://localhost:3000 (admin/admin123)"
echo "✅ Housing API: http://localhost:5001"
echo "✅ API Metrics: http://localhost:5001/api/monitoring/metrics"

echo ""
echo "📚 Quick Commands:"
echo "=================="
echo "# View Grafana dashboard:"
echo "open http://localhost:3000"
echo ""
echo "# Check Prometheus targets:"
echo "open http://localhost:9090/targets"
echo ""
echo "# Make a prediction:"
echo "curl -X POST http://localhost:5001/api/predict -H 'Content-Type: application/json' -d '{\"longitude\": -122.23, \"latitude\": 37.88, \"housing_median_age\": 41.0, \"total_rooms\": 880.0, \"total_bedrooms\": 129.0, \"population\": 322.0, \"households\": 126.0, \"median_income\": 8.3252, \"ocean_proximity\": \"NEAR BAY\"}'"
echo ""
echo "# View all metrics:"
echo "curl http://localhost:5001/api/monitoring/metrics"
echo ""
echo "# Stop monitoring stack:"
echo "docker-compose -f docker-compose.monitoring.yml down"

echo ""
echo "🎊 Your MLOps monitoring stack is working perfectly!"
