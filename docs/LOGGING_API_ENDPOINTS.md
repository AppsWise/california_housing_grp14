# 📊 **Prediction Logging API Endpoints**

## 🎯 **Overview**

Instead of writing SQLite queries manually, you can now use these convenient REST API endpoints to access all prediction logs, statistics, and analytics through simple HTTP requests.

## 🔗 **Available Endpoints**

### **1. 📋 Get All Prediction Logs**
```bash
GET /api/logs/predictions
```

**Query Parameters:**
- `limit` (int, default: 50) - Number of records per page
- `offset` (int, default: 0) - Pagination offset
- `hours` (int, default: 24) - Time window in hours
- `endpoint` (string, optional) - Filter by specific API endpoint
- `min_price` (float, optional) - Minimum prediction value
- `max_price` (float, optional) - Maximum prediction value

**Example Usage:**
```bash
# Get last 10 predictions
curl "http://localhost:5001/api/logs/predictions?limit=10"

# Get predictions above $300k in last 48 hours
curl "http://localhost:5001/api/logs/predictions?min_price=300000&hours=48"

# Get predictions from batch endpoint only
curl "http://localhost:5001/api/logs/predictions?endpoint=/api/predict/batch"

# Pagination - get next 20 records
curl "http://localhost:5001/api/logs/predictions?limit=20&offset=20"
```

**Response Format:**
```json
{
  "logs": [
    {
      "id": 5,
      "timestamp": "2025-08-06T01:36:31.982804",
      "endpoint": "/api/predict",
      "method": "POST",
      "input_data": {
        "longitude": -117.12,
        "latitude": 32.75,
        "housing_median_age": 15.0,
        "median_income": 7.5,
        "ocean_proximity": "NEAR OCEAN"
      },
      "prediction": 237400.0,
      "processing_time_ms": 9.88,
      "status_code": 200,
      "error_message": null
    }
  ],
  "pagination": {
    "total_count": 5,
    "limit": 50,
    "offset": 0,
    "has_more": false
  },
  "filters": {
    "hours": 24,
    "endpoint": null,
    "min_price": null,
    "max_price": null
  }
}
```

---

### **2. 📈 Get Prediction Statistics**
```bash
GET /api/logs/predictions/stats
```

**Query Parameters:**
- `hours` (int, default: 24) - Time window in hours
- `endpoint` (string, optional) - Filter by specific API endpoint

**Example Usage:**
```bash
# Get stats for last 24 hours
curl "http://localhost:5001/api/logs/predictions/stats"

# Get stats for last week
curl "http://localhost:5001/api/logs/predictions/stats?hours=168"

# Get stats for single predictions only
curl "http://localhost:5001/api/logs/predictions/stats?endpoint=/api/predict"
```

**Response Format:**
```json
{
  "statistics": {
    "total_predictions": 5,
    "successful_predictions": 5,
    "predictions_with_values": 5,
    "error_rate_percent": 0.0,
    "avg_processing_time_ms": 18.12,
    "avg_prediction_value": 245420.0,
    "min_prediction_value": 107500.0,
    "max_prediction_value": 371300.0,
    "time_window_hours": 24
  },
  "hourly_breakdown": [
    {
      "hour": "2025-08-06 01:00:00",
      "prediction_count": 3,
      "avg_prediction": 329333.33,
      "min_prediction": 237400.0,
      "max_prediction": 371300.0,
      "avg_processing_time_ms": 8.82
    }
  ]
}
```

---

### **3. 🕒 Get Recent Predictions**
```bash
GET /api/logs/predictions/recent
```

**Query Parameters:**
- `count` (int, default: 10) - Number of recent predictions

**Example Usage:**
```bash
# Get last 5 predictions
curl "http://localhost:5001/api/logs/predictions/recent?count=5"

# Get last prediction
curl "http://localhost:5001/api/logs/predictions/recent?count=1"
```

**Response Format:**
```json
{
  "recent_predictions": [
    {
      "id": 5,
      "timestamp": "2025-08-06T01:36:31.982804",
      "endpoint": "/api/predict",
      "input_data": {
        "longitude": -117.12,
        "latitude": 32.75,
        "median_income": 7.5
      },
      "prediction": 237400.0,
      "processing_time_ms": 9.88
    }
  ],
  "count": 1
}
```

---

### **4. 🔍 Search Predictions by Input Features**
```bash
GET /api/logs/predictions/search
```

**Query Parameters (Housing Features):**
- `longitude` (float) - Longitude coordinate
- `latitude` (float) - Latitude coordinate  
- `median_income` (float) - Median income value
- `housing_median_age` (float) - Median age of housing
- `total_rooms` (int) - Total number of rooms
- `total_bedrooms` (int) - Total number of bedrooms
- `population` (int) - Population count
- `households` (int) - Number of households
- `ocean_proximity` (string) - Ocean proximity category
- `limit` (int, default: 20) - Max results to return

**Example Usage:**
```bash
# Find predictions for San Francisco area
curl "http://localhost:5001/api/logs/predictions/search?longitude=-122.23&latitude=37.88"

# Find predictions for high-income areas
curl "http://localhost:5001/api/logs/predictions/search?median_income=8.3252"

# Find predictions for inland properties
curl "http://localhost:5001/api/logs/predictions/search?ocean_proximity=INLAND"

# Complex search with multiple criteria
curl "http://localhost:5001/api/logs/predictions/search?longitude=-117.12&median_income=7.5&limit=10"
```

**Response Format:**
```json
{
  "search_results": [
    {
      "id": 3,
      "timestamp": "2025-08-06T01:06:21.376705",
      "endpoint": "/api/predict",
      "input_data": {
        "longitude": -122.23,
        "latitude": 37.88,
        "median_income": 8.3252
      },
      "prediction": 371300.0,
      "processing_time_ms": 9.33
    }
  ],
  "search_criteria": {
    "longitude": -122.23,
    "latitude": 37.88
  },
  "count": 2
}
```

---

## 🚀 **Real-World Usage Examples**

### **1. Monitor Model Performance**
```bash
# Check error rates and response times
curl "http://localhost:5001/api/logs/predictions/stats" | jq '.statistics'

# Monitor hourly prediction volumes
curl "http://localhost:5001/api/logs/predictions/stats" | jq '.hourly_breakdown'
```

### **2. Debug Specific Issues**
```bash
# Find recent errors
curl "http://localhost:5001/api/logs/predictions?limit=100" | \
  jq '.logs[] | select(.status_code != 200)'

# Find slow predictions
curl "http://localhost:5001/api/logs/predictions" | \
  jq '.logs[] | select(.processing_time_ms > 50)'
```

### **3. Business Analytics**
```bash
# Find high-value predictions
curl "http://localhost:5001/api/logs/predictions?min_price=500000" | \
  jq '.logs[].prediction'

# Compare predictions by location
curl "http://localhost:5001/api/logs/predictions/search?ocean_proximity=NEAR%20OCEAN"
```

### **4. Data Validation**
```bash
# Check recent prediction accuracy
curl "http://localhost:5001/api/logs/predictions/recent?count=20" | \
  jq '.recent_predictions[] | {input: .input_data.median_income, prediction: .prediction}'

# Find duplicate predictions
curl "http://localhost:5001/api/logs/predictions/search?longitude=-122.23&latitude=37.88"
```

---

## 📊 **Response Time & Pagination**

### **Performance**
- All endpoints return `processing_time_ms` for performance monitoring
- Typical response times: 0.5-5ms for most queries
- Database queries are optimized with indexes

### **Pagination**
- Use `limit` and `offset` for large datasets
- `has_more` field indicates if more records exist
- `total_count` shows total matching records

### **Filtering**
- Combine multiple filters for precise queries
- Time-based filtering with `hours` parameter
- Price range filtering with `min_price`/`max_price`
- Endpoint-specific filtering

---

## 🔧 **Integration Examples**

### **Python Client**
```python
import requests
import json

# Get recent predictions
response = requests.get('http://localhost:5001/api/logs/predictions/recent?count=5')
recent_preds = response.json()['recent_predictions']

# Search by location
search_params = {'longitude': -122.23, 'latitude': 37.88}
response = requests.get('http://localhost:5001/api/logs/predictions/search', params=search_params)
location_preds = response.json()['search_results']

# Get performance stats
response = requests.get('http://localhost:5001/api/logs/predictions/stats')
stats = response.json()['statistics']
print(f"Average response time: {stats['avg_processing_time_ms']}ms")
```

### **JavaScript/Node.js**
```javascript
// Fetch recent predictions
const response = await fetch('http://localhost:5001/api/logs/predictions/recent?count=10');
const data = await response.json();
console.log('Recent predictions:', data.recent_predictions);

// Search predictions
const searchUrl = new URL('http://localhost:5001/api/logs/predictions/search');
searchUrl.searchParams.set('median_income', '8.0');
const searchResponse = await fetch(searchUrl);
const searchData = await searchResponse.json();
```

### **curl + jq Scripts**
```bash
#!/bin/bash
# Monitor prediction quality

echo "📊 Prediction Statistics (Last 24h)"
curl -s "http://localhost:5001/api/logs/predictions/stats" | \
  jq '.statistics | {
    total: .total_predictions,
    avg_price: .avg_prediction_value,
    avg_time: .avg_processing_time_ms,
    error_rate: .error_rate_percent
  }'

echo "🏠 Recent High-Value Predictions"
curl -s "http://localhost:5001/api/logs/predictions?min_price=400000&limit=5" | \
  jq '.logs[] | {
    timestamp: .timestamp,
    location: [.input_data.longitude, .input_data.latitude],
    price: .prediction
  }'
```

---

## ✅ **Benefits Over Direct SQLite Queries**

| **Feature** | **SQLite Queries** | **API Endpoints** |
|-------------|-------------------|-------------------|
| **Ease of Use** | ❌ Complex SQL syntax | ✅ Simple HTTP requests |
| **Filtering** | ❌ Manual WHERE clauses | ✅ URL parameters |
| **Pagination** | ❌ Manual LIMIT/OFFSET | ✅ Built-in pagination |
| **JSON Parsing** | ❌ Manual parsing needed | ✅ Structured JSON response |
| **Error Handling** | ❌ Raw database errors | ✅ Formatted error messages |
| **Performance** | ❌ Direct DB access needed | ✅ Optimized queries |
| **Integration** | ❌ Database driver required | ✅ Standard HTTP client |
| **Security** | ❌ Direct DB exposure | ✅ API access control |

Now you have **convenient REST API endpoints** instead of writing SQLite queries! 🎉
