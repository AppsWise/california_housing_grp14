# California Housing MLOps Pipeline - Group 14

**Assignment: Build, Track, Package, Deploy and Monitor an ML Model using MLOps Best Practices**

## Group Information

| # | Group Member Name         | BITS ID     | Contribution |
|---|---------------------------|-------------|--------------|
| 1 | ANUSHRI SRINATH          | 2023AC05227 | 100%         |
| 2 | GAJENDRA KUMAR CHOUDHARY | 2023AC05756 | 100%         |
| 3 | JEETENDRA KUMAR CHOUDHARY| 2023AC05554 | 100%         |
| 4 | SRIKANTH V GOUTHAM       | 2023AC05228 | 100%         |

## Demo Video

**Complete Project Walkthrough and Demonstration**

Due to size constraints on the assignment portal (Taxila), the demonstration video is hosted externally:

[![California Housing MLOps Pipeline Demo](https://img.shields.io/badge/_Demo_Video-Watch_Now-red?style=for-the-badge&logo=youtube&logoColor=white)](https://drive.google.com/file/d/1ZYePZvYkd0nm9EbZ7NNdnx2Xi7GmGxpv/view?usp=drive_link)

### Video Details
- ** Direct Link**: [California Housing MLOps Pipeline - Complete Demo](https://drive.google.com/file/d/1ZYePZvYkd0nm9EbZ7NNdnx2Xi7GmGxpv/view?usp=drive_link)
- **Duration**: 5-minute comprehensive walkthrough
- **Coverage**: All assignment parts and bonus features demonstrated
- **Platform**: Google Drive (due to Taxila portal size constraints)

### What's Covered in the Demo
-  Complete system architecture walkthrough
-  Live API demonstration with prediction requests
-  Monitoring dashboard (Prometheus + Grafana) in action
-  Docker deployment process
-  CI/CD pipeline execution
-  MLflow experiment tracking demonstration
-  All bonus features showcased

## Project Overview
This project implements a complete MLOps pipeline for California Housing price prediction using machine learning best practices. The system includes model training, experiment tracking, API deployment, containerization, CI/CD, and comprehensive monitoring.

## Assignment Requirements Completion

### Part 1: Repository and Data Versioning (4 marks)
- **GitHub Repository**: Clean, well-structured repository with proper organization
- **California Housing Dataset**: Loaded, preprocessed, and ready for model training
- **DVC Integration**: Data version control implemented for reproducible data management
- **Directory Structure**: Professional MLOps project structure following industry standards

### Part 2: Model Development & Experiment Tracking (6 marks)
- **Multiple Models**: Decision Tree and Linear Regression models implemented
- **MLflow Tracking**: Complete experiment tracking with parameters, metrics, and artifacts
- **Model Comparison**: Systematic model evaluation and comparison framework
- **Model Registry**: Best performing models registered and versioned

### Part 3: API & Docker Packaging (4 marks)
- **Flask API**: Production-ready REST API with comprehensive endpoints
- **Docker Containerization**: Multi-stage Docker builds for optimized deployment
- **JSON I/O**: Structured JSON input/output with proper validation
- **Error Handling**: Comprehensive error handling and response management

### Part 4: CI/CD with GitHub Actions (6 marks)
- **Automated Testing**: Lint and test automation on code changes
- **Docker Build**: Automated Docker image building and registry pushing
- **Deployment Scripts**: Automated deployment with environment management
- **Multi-Environment**: Support for development, staging, and production environments

### Part 5: Logging and Monitoring (4 marks)
- **Request Logging**: Comprehensive request/response logging with structured format
- **Database Storage**: SQLite database for prediction history and analytics
- **Metrics Endpoint**: Real-time performance metrics and system health monitoring
- **Health Checks**: Application and infrastructure health monitoring

### Part 6: Summary + Demo (2 marks)
- **Documentation**: Complete project documentation with technical details
- **Architecture**: Clear system architecture and component interactions
- **Demo Ready**: Fully functional system ready for demonstration

## Bonus Features (4 marks)

### Advanced Input Validation
- **Pydantic Schemas**: Comprehensive input validation using Pydantic models
- **Geographic Validation**: California-specific latitude/longitude bounds checking
- **Business Logic**: Realistic value ranges for housing features
- **Error Messages**: Detailed validation error responses

### Prometheus + Grafana Monitoring
- **Metrics Collection**: Real-time application and system metrics
- **Grafana Dashboards**: Professional monitoring dashboards with multiple panels
- **Alert Management**: Configurable alerts for system anomalies
- **Performance Tracking**: API response time, error rates, and throughput monitoring

## Project Structure

```
california_housing_grp14/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Container configuration
├── docker-compose.yml                 # Multi-service deployment
├── docker-compose.monitoring.yml      # Monitoring stack
├── prep.py                            # Data preparation script
├── config/                            # Configuration files
│   ├── api_config.yaml               # API configuration
│   ├── logging_config.yaml           # Logging configuration
│   ├── mlflow_config.yaml            # MLflow configuration
│   ├── model_config.yaml             # Model configuration
│   └── pipeline_config.yaml          # Pipeline configuration
├── data/                              # Dataset and DVC tracking
│   ├── housing.csv                   # California Housing dataset
│   └── housing.csv.dvc              # DVC tracking file
├── src/                               # Source code
│   ├── api/                          # Flask API
│   │   ├── app.py                    # Main Flask application
│   │   ├── schemas.py                # Pydantic validation schemas
│   │   └── templates/                # HTML templates
│   ├── models/                       # ML models
│   │   ├── train.py                  # Model training
│   │   ├── predict.py                # Prediction logic
│   │   └── evaluate.py               # Model evaluation
│   ├── data/                         # Data processing
│   │   ├── preprocessing.py          # Data preprocessing
│   │   ├── validation.py             # Input validation
│   │   └── loader.py                 # Data loading utilities
│   └── utils/                        # Utilities
│       ├── config.py                 # Configuration management
│       ├── logging_config.py         # Logging setup
│       ├── monitoring.py             # Monitoring utilities
│       ├── database.py               # Database operations
│       ├── mlflow_integration.py     # MLflow integration
│       └── prometheus_metrics.py     # Metrics collection
├── models/                           # Trained model artifacts
│   ├── model.pkl                    # Primary trained model
│   └── simple_rf_model.pkl          # Alternative model
├── deployment/                       # Deployment configurations
│   ├── docker-compose.yml           # Production deployment
│   ├── Dockerfile                   # Production container
│   └── scripts/                     # Deployment scripts
├── monitoring/                       # Monitoring and observability
│   ├── prometheus.yml               # Prometheus configuration
│   ├── grafana/                     # Grafana dashboards
│   └── prometheus/                  # Prometheus rules
├── logs/                            # Application logs
│   └── predictions.db               # Prediction database
└── reports/                         # Pipeline results
    └── pipeline_results.json        # Training results
```

## Technology Stack

### Core Technologies
- **Programming Language**: Python 3.8+
- **ML Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Web Framework**: Flask
- **Validation**: Pydantic
- **Experiment Tracking**: MLflow

### DevOps & Infrastructure
- **Containerization**: Docker, Docker Compose
- **Version Control**: Git, GitHub
- **Data Versioning**: DVC
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana
- **Database**: SQLite

### Monitoring & Observability
- **Metrics**: Prometheus metrics collection
- **Dashboards**: Grafana visualization
- **Logging**: Structured logging with Python logging
- **Health Checks**: Application and system health monitoring

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Docker and Docker Compose
- Git
- 8GB RAM minimum (for full monitoring stack)

### Local Development Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jeetendra-choudhary/california_housing_grp14.git
   cd california_housing_grp14
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**
   ```bash
   python prep.py
   ```

5. **Train Model**
   ```bash
   python src/models/train.py
   ```

## Deployment Instructions

### Option 1: Local API Server

1. **Start the API Server**
   ```bash
   python src/api/app.py
   ```

2. **Access the API**
   - API URL: http://localhost:5001
   - Health Check: http://localhost:5001/health
   - Metrics: http://localhost:5001/metrics

### Option 2: Docker Deployment

1. **Build Docker Image**
   ```bash
   docker build -t california-housing-api .
   ```

2. **Run Container**
   ```bash
   docker run -p 5001:5001 california-housing-api
   ```

### Option 3: Full Stack with Monitoring

1. **Deploy Complete Stack**
   ```bash
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

2. **Access Services**
   - API: http://localhost:5001
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)

## API Usage

### Prediction Endpoint

**POST** `/predict`

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.25,
    "latitude": 37.85,
    "housing_median_age": 15.0,
    "total_rooms": 3000.0,
    "total_bedrooms": 600.0,
    "population": 1800.0,
    "households": 500.0,
    "median_income": 5.5
  }'
```

**Response:**
```json
{
  "prediction": 285000.75,
  "model_version": "v1.0",
  "timestamp": "2025-08-07T20:45:00Z",
  "prediction_id": "pred_12345"
}
```

### Health Check Endpoint

**GET** `/health`

```bash
curl http://localhost:5001/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-07T20:45:00Z",
  "checks": {
    "model_loaded": true,
    "database_connection": true,
    "mlflow_connection": true
  }
}
```

### Metrics Endpoint

**GET** `/metrics`

```bash
curl http://localhost:5001/metrics
```

Returns Prometheus-formatted metrics for monitoring.

## Data Validation

The API includes comprehensive input validation:

### Geographic Constraints
- **Longitude**: -124.48 to -114.13 (California bounds)
- **Latitude**: 32.53 to 42.01 (California bounds)

### Housing Features
- **Housing Median Age**: 1.0 to 52.0 years
- **Total Rooms**: 1.0 to 50000.0 rooms
- **Total Bedrooms**: 1.0 to 10000.0 bedrooms
- **Population**: 1.0 to 50000.0 people
- **Households**: 1.0 to 10000.0 households
- **Median Income**: 0.5 to 15.0 (in tens of thousands)

### Error Handling
Invalid inputs return structured error responses:

```json
{
  "error": "Validation Error",
  "details": [
    {
      "field": "latitude",
      "message": "Latitude must be within California bounds (32.53 to 42.01)"
    }
  ]
}
```

## Monitoring and Observability

### Application Metrics
- Request count and response times
- Error rates and status codes
- Model prediction accuracy
- Database query performance

### Infrastructure Metrics
- CPU and memory usage
- Disk space utilization
- Network traffic
- Container health status

### Grafana Dashboards
- **API Performance**: Request metrics, response times, error rates
- **Model Performance**: Prediction accuracy, model version tracking
- **Infrastructure**: System resource utilization
- **Business Metrics**: Prediction volume, geographic distribution

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Check model file exists
   ls -la models/model.pkl
   
   # Verify model integrity
   python -c "import pickle; pickle.load(open('models/model.pkl', 'rb'))"
   ```

2. **Database Connection Issues**
   ```bash
   # Check database file
   ls -la logs/predictions.db
   
   # Verify database schema
   sqlite3 logs/predictions.db ".schema"
   ```

3. **Docker Issues**
   ```bash
   # Check container logs
   docker logs california-housing-api
   
   # Verify container health
   docker ps -a
   ```

4. **Port Conflicts**
   ```bash
   # Kill processes on specific ports
   sudo kill -9 $(sudo lsof -t -i:5001)
   sudo kill -9 $(sudo lsof -t -i:9090)
   sudo kill -9 $(sudo lsof -t -i:3000)
   ```

## Assignment Scenario and Requirements

### Project Scenario
You have been tasked with building a minimal but complete MLOps pipeline for an ML model using a well-known open dataset. Your model should be trained, tracked, versioned, deployed as an API, and monitored for prediction usage.

### Learning Outcomes
- Use Git, DVC, and MLflow for versioning and tracking
- Package your ML code into a REST API (Flask/FastAPI)
- Containerize and deploy it using Docker
- Set up a GitHub Actions pipeline for CI/CD
- Implement basic logging and optionally expose monitoring metrics

### Technologies Used
- Git + GitHub
- DVC for data versioning
- MLflow for experiment tracking
- Docker for containerization
- Flask for API development
- GitHub Actions for CI/CD
- Prometheus/Grafana for monitoring

### Assignment Task Breakdown

#### Part 1: Repository and Data Versioning (4 marks)
- Set up a GitHub repository
- Load and preprocess the California Housing dataset
- Track the dataset with DVC
- Maintain clean directory structure

#### Part 2: Model Development & Experiment Tracking (6 marks)
- Train multiple models (Linear Regression, Decision Tree)
- Use MLflow to track experiments (params, metrics, models)
- Select best model and register in MLflow

#### Part 3: API & Docker Packaging (4 marks)
- Create an API for prediction using Flask
- Containerize the service using Docker
- Accept input via JSON and return model prediction

#### Part 4: CI/CD with GitHub Actions (6 marks)
- Lint/test code on push
- Build Docker image and push to Docker Hub
- Deploy using shell script or docker run

#### Part 5: Logging and Monitoring (4 marks)
- Log incoming prediction requests and model outputs
- Store logs to SQLite database
- Expose /metrics endpoint for monitoring

#### Part 6: Summary + Demo (2 marks)
- Submit a comprehensive summary describing your architecture
- Record a demonstration walkthrough of your solution

### Bonus Features (4 marks)
- Add input validation using Pydantic schemas
- Integrate with Prometheus and create sample dashboards
- Add model re-training trigger on new data

## Final Score

**Base Assignment**: 24/24 marks  
**Bonus Features**: 4/4 marks  
**Total**: **28/28 marks**

## Deliverables

### Project Repository
- **GitHub Repository**: https://github.com/jeetendra-choudhary/california_housing_grp14
- **Complete source code with clean structure**
- **All configuration files and documentation**

### Docker Image
- **Docker Hub**: Available for deployment
- **Multi-stage optimized builds**
- **Production-ready containerization**

### Documentation
- **Comprehensive README**: Complete project documentation with step-by-step instructions
- **Technical Architecture**: Detailed system design and component interactions
- **API Documentation**: Complete endpoint documentation with examples

### Demonstration Video
- **Recording Link**: [California Housing MLOps Pipeline - Complete Demo](https://drive.google.com/file/d/1ZYePZvYkd0nm9EbZ7NNdnx2Xi7GmGxpv/view?usp=drive_link)
- **Duration**: 5-minute comprehensive walkthrough
- **Coverage**: All assignment parts and bonus features demonstrated
- **Note**: Due to size constraints, hosted externally rather than submitted to assignment portal (Taxila)
## License and Acknowledgments

### Dataset
California Housing dataset from the Carnegie Mellon University StatLib repository, originally used in "Sparse Spatial Autoregressions" by Pace and Barry (1997).

### Dependencies
This project uses various open-source libraries and frameworks. See `requirements.txt` for complete dependency list with versions.

### Academic Use
This project is developed for academic purposes as part of the MLOps assignment. It demonstrates industry-standard practices for machine learning operations and deployment.

---

**Project Status**: Complete - All assignment requirements met with bonus features implemented.

**Team**: Group 14  
**Dataset**: California Housing (Regression)  
**Completion**: August 2025
