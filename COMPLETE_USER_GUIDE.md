# 🏠 California Housing Price Prediction - Complete MLOps Guide

**A comprehensive end-to-end MLOps pipeline for predicting California housing prices using Machine Learning, MLflow, Docker, and CI/CD automation.**

---

## 📋 Table of Contents

1. [🎯 Project Overview](#-project-overview)
2. [🚀 Quick Start Guide](#-quick-start-guide)
3. [📁 Project Structure](#-project-structure)
4. [🔧 Setup & Installation](#-setup--installation)
5. [🤖 Model Training & MLflow](#-model-training--mlflow)
6. [🌐 API Development](#-api-development)
7. [🐳 Docker Deployment](#-docker-deployment)
8. [⚙️ CI/CD Pipeline](#️-cicd-pipeline)
9. [🧪 Testing](#-testing)
10. [📊 Monitoring](#-monitoring)
11. [📚 API Reference](#-api-reference)
12. [🛠️ Troubleshooting](#️-troubleshooting)

---

## 🎯 Project Overview

This project implements a complete MLOps pipeline for California housing price prediction, demonstrating industry best practices for machine learning operations.

### 🎯 **Project Goals Achieved**

✅ **Part 1**: Repository Setup & Data Versioning with DVC  
✅ **Part 2**: Model Development & Experiment Tracking with MLflow  
✅ **Part 3**: API Development & Docker Packaging  
✅ **Part 4**: CI/CD Pipeline Implementation with GitHub Actions  
✅ **Part 5**: Project Restructuring & Comprehensive Documentation  

---

## 🚀 Quick Start Guide

### 🎯 **30-Second Demo**

```bash
# 1. Clone & Setup
git clone <repository-url>
cd california_housing_mlops
pip install -r requirements.txt

# 2. Train Model
python src/models/train.py

# 3. Start API
python src/api/app.py

# 4. Test Prediction
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"longitude":-122.23,"latitude":37.88,"housing_median_age":41.0,"total_rooms":880.0,"total_bedrooms":129.0,"population":322.0,"households":126.0,"median_income":8.3252,"ocean_proximity":"NEAR BAY"}'
```

### 🐳 **Docker Quick Start**

```bash
# One-command deployment
cd deployment && ./scripts/build.sh

# Access the application
curl http://localhost:5001/health
```

---

## 📁 Project Structure

```
california_housing_mlops/
├── 📄 README.md                    # Quick start guide
├── 📄 COMPLETE_USER_GUIDE.md       # This comprehensive guide
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore patterns
│
├── 📁 data/                        # Data storage and versioning
│   ├── 📄 housing.csv             # California housing dataset
│   └── 📄 housing.csv.dvc         # DVC tracking file
│
├── 📁 src/                         # Source code organization
│   ├── 📁 models/                 # Model training and evaluation
│   │   ├── 📄 __init__.py         # Package initialization
│   │   └── 📄 train.py            # MLflow model training script
│   │
│   ├── 📁 api/                    # Flask API application
│   │   ├── 📄 __init__.py         # Package initialization
│   │   ├── 📄 app.py              # Flask API with health/metrics
│   │   └── 📁 templates/          # HTML templates
│   │       └── 📄 index.html      # Web interface
│   │
│   └── 📁 utils/                  # Utility functions
│       ├── 📄 __init__.py         # Package initialization
│       └── 📄 data_processing.py  # Data processing utilities
│
├── 📁 models/                      # Trained model artifacts
│   └── 📄 model.pkl               # Best trained model
│
├── 📁 notebooks/                   # Jupyter notebooks
│   └── 📄 eda.ipynb               # Exploratory data analysis
│
├── 📁 tests/                       # Comprehensive testing
│   ├── 📄 __init__.py             # Package initialization
│   ├── 📄 conftest.py             # Test configuration & fixtures
│   ├── 📄 test_api.py             # API endpoint testing
│   └── 📄 test_models.py          # Model functionality testing
│
├── 📁 deployment/                  # Docker and deployment
│   ├── 📄 Dockerfile              # Production-ready container
│   ├── 📄 docker-compose.yml      # Multi-service orchestration
│   ├── 📁 scripts/                # Deployment automation
│   │   ├── 📄 build.sh            # Linux/macOS build script
│   │   └── 📄 build.ps1           # Windows PowerShell script
│   │
│   └── 📁 monitoring/             # Monitoring configuration
│       └── 📄 prometheus.yml      # Prometheus metrics config
│
├── 📁 .github/workflows/          # CI/CD automation
│   ├── 📄 lint.yml               # Code quality & testing
│   └── 📄 build-and-push.yml     # Docker build & registry push
│
└── 📁 config/                     # Configuration management
    ├── 📄 model_config.yaml       # Model training parameters
    └── 📄 api_config.yaml         # API server configuration
```

---

## 🔧 Setup & Installation

### 📋 **Prerequisites**

- **Python 3.11+** (recommended)
- **Git** for version control
- **Docker** for containerization
- **DVC** for data versioning

### 🛠️ **Installation Steps**

#### 1. **Clone Repository**
```bash
git clone <repository-url>
cd california_housing_mlops
```

#### 2. **Set Up Python Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. **Data Setup**
```bash
# Pull data with DVC
dvc pull

# Verify data
ls data/
# Should show: housing.csv, housing.csv.dvc
```

#### 4. **Verify Installation**
```bash
# Run tests
pytest tests/

# Check MLflow (optional)
python check_mlflow.py
```

---

## 🤖 Model Training & MLflow

### 🎯 **Model Overview**

Our project implements and compares two machine learning models:

| Model | RMSE | R² Score | Status |
|-------|------|----------|--------|
| **Decision Tree Regressor** | ~72,789 | ~0.596 | ✅ **Best Model** |
| **Linear Regression** | ~83,858 | ~0.463 | Baseline |

### 🔬 **Training Process**

#### 1. **Train Models**
```bash
# Run training with MLflow tracking
python src/models/train.py
```

**What happens during training:**
- ✅ Data loading and preprocessing
- ✅ Feature engineering (rooms per household, etc.)
- ✅ Model training (Decision Tree & Linear Regression)
- ✅ Performance evaluation and comparison
- ✅ MLflow experiment tracking
- ✅ Best model selection and registration
- ✅ Model artifact saving

#### 2. **View Results**
```bash
# Start MLflow UI (optional)
mlflow ui --port 5000

# Visit http://localhost:5000 to see experiments
```

### 📊 **Model Features**

The model uses these input features:
- **longitude** - Longitude coordinate
- **latitude** - Latitude coordinate  
- **housing_median_age** - Median age of houses
- **total_rooms** - Total number of rooms
- **total_bedrooms** - Total number of bedrooms
- **population** - Population in the area
- **households** - Number of households
- **median_income** - Median income (in tens of thousands)
- **ocean_proximity** - Categorical (NEAR BAY, <1H OCEAN, INLAND, NEAR OCEAN, ISLAND)

---

## 🌐 API Development

### 🎯 **API Overview**

Our Flask API provides RESTful endpoints for model serving with comprehensive monitoring and error handling.

### 🚀 **Starting the API**

```bash
# Start API server
python src/api/app.py

# API will be available at: http://localhost:5001
```

### 📋 **Available Endpoints**

| Endpoint | Method | Purpose | Response |
|----------|--------|---------|----------|
| `/health` | GET | Health check | Status information |
| `/api/predict` | POST | Single prediction | Prediction value |
| `/api/predict/batch` | POST | Batch predictions | Array of predictions |
| `/metrics` | GET | Prometheus metrics | Metrics data |
| `/` | GET | Web interface | HTML page |

### 🔍 **Testing the API**

#### **Health Check**
```bash
curl http://localhost:5001/health
```

#### **Single Prediction**
```bash
curl -X POST http://localhost:5001/api/predict \
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
  }'
```

---

## 🐳 Docker Deployment

### 🎯 **Container Overview**

Our Docker setup includes:
- **Optimized Dockerfile** with Python 3.11-slim
- **Security best practices** (non-root user, minimal attack surface)
- **Health checks** and monitoring
- **Multi-service orchestration** with docker-compose

### 🚀 **Quick Deployment**

#### **Docker Compose (Recommended)**
```bash
cd deployment
./scripts/build.sh
```

This starts:
- **API** on port 5001
- **Prometheus** on port 9090  
- **Grafana** on port 3000

#### **Manual Docker Build**
```bash
# Build image
docker build -f deployment/Dockerfile -t california-housing .

# Run container
docker run -d --name housing-api -p 5001:5000 california-housing

# Check health
curl http://localhost:5001/health
```

### 🔍 **Container Management**

```bash
# Check running containers
docker ps

# View logs
docker logs california-housing-api

# Stop services
cd deployment && docker-compose down

# Restart services
cd deployment && docker-compose restart
```

---

## ⚙️ CI/CD Pipeline

### 🎯 **Pipeline Overview**

Our GitHub Actions CI/CD pipeline includes 2 comprehensive workflows:

### 📋 **Workflow Details**

#### 1. **🔍 Code Quality & Testing** (`lint.yml`)
**Triggers:** Push/PR to main, develop
```yaml
- Flake8 linting
- Unit tests with pytest
- Docker file validation
- Security scanning
- Code coverage reporting
```

#### 2. **🏗️ Build & Deploy** (`build-and-push.yml`)
**Triggers:** Push to main (after tests pass)
```yaml
- Docker image building
- Push to Docker Hub
- Integration testing
- Staging deployment
```

### 🔧 **Setup CI/CD**

#### 1. **GitHub Secrets Required**
```
DOCKERHUB_USERNAME - Your Docker Hub username
DOCKERHUB_TOKEN - Your Docker Hub access token
```

#### 2. **Workflow Configuration**
The workflows are pre-configured and will automatically:
- ✅ Run on code changes
- ✅ Build and test Docker images  
- ✅ Deploy to staging
- ✅ Run integration tests

---

## 🧪 Testing

### 🎯 **Testing Strategy**

Our comprehensive testing includes:

| Test Type | Files | Coverage |
|-----------|-------|----------|
| **Unit Tests** | `test_models.py` | Model functionality |
| **API Tests** | `test_api.py` | Endpoint validation |
| **Integration Tests** | GitHub Actions | End-to-end workflows |

### 🚀 **Running Tests**

#### **Local Testing**
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

#### **API Testing**
```bash
# Test health endpoint
curl http://localhost:5001/health

# Test prediction endpoint
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"longitude":-122.23,"latitude":37.88,"housing_median_age":41.0,"total_rooms":880.0,"total_bedrooms":129.0,"population":322.0,"households":126.0,"median_income":8.3252,"ocean_proximity":"NEAR BAY"}'
```

---

## 📊 Monitoring

### 🎯 **Monitoring Stack**

| Component | Port | Purpose |
|-----------|------|---------|
| **Prometheus** | 9090 | Metrics collection |
| **Grafana** | 3000 | Dashboard visualization |
| **Flask Metrics** | 5001/metrics | Application metrics |

### 🚀 **Available Metrics**

#### **Application Metrics**
- **Request Count** - Total API requests
- **Request Duration** - Response time histogram
- **Prediction Count** - Number of predictions made
- **Error Rate** - Failed request percentage
- **Health Status** - Service availability

### 🔍 **Accessing Monitoring**

#### **Prometheus**
```bash
# Start monitoring stack
cd deployment
docker-compose up -d

# Access Prometheus
open http://localhost:9090

# Sample queries:
# - flask_http_request_total
# - flask_http_request_duration_seconds
```

#### **Grafana**
```bash
# Access Grafana
open http://localhost:3000

# Default credentials:
# Username: admin
# Password: admin
```

---

## 📚 API Reference

### 🎯 **Complete API Documentation**

#### **Base URL**
```
http://localhost:5001
```

### 📋 **Endpoint Reference**

#### **1. Health Check**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00Z",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### **2. Single Prediction**
```http
POST /api/predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41.0,
  "total_rooms": 880.0,
  "total_bedrooms": 129.0,
  "population": 322.0,
  "households": 126.0,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}
```

**Response:**
```json
{
  "prediction": 452600.123,
  "model_version": "1.0.0",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

#### **3. Batch Prediction**
```http
POST /api/predict/batch
Content-Type: application/json
```

**Request Body:**
```json
[
  {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }
]
```

**Response:**
```json
{
  "predictions": [452600.123],
  "count": 1,
  "model_version": "1.0.0",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

### ❌ **Error Responses**

#### **400 Bad Request**
```json
{
  "error": "Invalid input data",
  "message": "Missing required field: longitude"
}
```

#### **500 Internal Server Error**
```json
{
  "error": "Model prediction failed",
  "message": "Internal server error"
}
```

---

## 🛠️ Troubleshooting

### 🚨 **Common Issues & Solutions**

#### **1. Model Loading Issues**

**Problem:** Model file not found
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/model.pkl'
```

**Solution:**
```bash
# Train the model first
python src/models/train.py

# Verify model exists
ls models/model.pkl
```

#### **2. Port Already in Use**

**Problem:** Port 5001 is already in use
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Find process using port
lsof -i :5001

# Kill the process
kill -9 <PID>

# Or use different port
export FLASK_PORT=5002
python src/api/app.py
```

#### **3. Docker Build Issues**

**Problem:** Docker build fails
```
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```

**Solution:**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild with no cache
docker build --no-cache -f deployment/Dockerfile -t california-housing .
```

#### **4. Test Failures**

**Problem:** Tests fail due to missing dependencies
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```bash
# Install in development mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### 🔍 **Debugging Commands**

#### **Check Service Status**
```bash
# API health
curl http://localhost:5001/health

# Docker containers
docker ps -a

# Container logs
docker logs california-housing-api

# Resource usage
docker stats
```

#### **Validate Configuration**
```bash
# Test model loading
python -c "import pickle; model = pickle.load(open('models/model.pkl', 'rb')); print('✅ Model loaded')"

# Test API imports
python -c "from src.api.app import app; print('✅ API imports working')"

# Validate Docker Compose
cd deployment && docker-compose config
```

---

## 🎉 Conclusion

You now have a complete, production-ready MLOps pipeline! This guide covers everything from basic setup to advanced deployment.

### 🚀 **What You've Built**

✅ **Complete MLOps Pipeline** - From data to deployment  
✅ **Automated CI/CD** - GitHub Actions workflows  
✅ **Containerized Application** - Docker deployment  
✅ **Monitoring & Observability** - Prometheus & Grafana  
✅ **Comprehensive Testing** - Unit, integration, and API tests  
✅ **Professional Documentation** - Complete guides and references  

### 🎯 **Next Steps**

1. **Deploy to Cloud** - AWS, GCP, or Azure
2. **Add Authentication** - Secure your APIs
3. **Scale Horizontally** - Load balancing and auto-scaling
4. **Advanced ML** - A/B testing, ensemble models
5. **Data Engineering** - Real-time pipelines, feature stores

---

**🎯 Happy MLOps! 🚀**

*This project demonstrates industry-standard MLOps practices. Feel free to adapt and extend it for your specific use cases.*
