# California Housing Price Prediction - MLOps Project

A complete MLOps pipeline for California Housing price prediction using scikit-learn, MLflow, Docker, and GitHub Actions.

## 📚 **Complete Documentation**

👉 **[VIEW COMPLETE USER GUIDE](COMPLETE_USER_GUIDE.md)** 👈

*For comprehensive setup, usage, API reference, deployment instructions, and troubleshooting, please refer to the complete user guide above.*

---

## 🚀 Quick Start

### 30-Second Demo
```bash
# 1. Setup
pip install -r requirements.txt

# 2. Train Model
python src/models/train.py

# 3. Start API
python src/api/app.py

# 4. Test
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"longitude":-122.23,"latitude":37.88,"housing_median_age":41.0,"total_rooms":880.0,"total_bedrooms":129.0,"population":322.0,"households":126.0,"median_income":8.3252,"ocean_proximity":"NEAR BAY"}'
```

### Docker Deployment
```bash
cd deployment && ./scripts/build.sh
curl http://localhost:5001/health
```

---

## 📁 Project Structure

```
california_housing_mlops/
├── 📄 README.md                    # This file - Quick start guide
├── 📄 COMPLETE_USER_GUIDE.md       # 📚 Complete documentation
├── 📄 requirements.txt             # Python dependencies
│
├── 📁 data/                        # Data storage with DVC tracking
├── 📁 src/                         # Source code (models, api, utils)
├── 📁 models/                      # Trained model artifacts
├── 📁 notebooks/                   # Jupyter notebooks
├── 📁 tests/                       # Comprehensive test suite
├── 📁 deployment/                  # Docker and deployment configs
├── 📁 .github/workflows/          # CI/CD automation
└── 📁 config/                      # Configuration management
```

---

## ✨ Features

- ✅ **Data Versioning** with DVC
- ✅ **Experiment Tracking** with MLflow
- ✅ **Model Training** with scikit-learn
- ✅ **REST API** with Flask
- ✅ **Containerization** with Docker
- ✅ **CI/CD Pipeline** with GitHub Actions
- ✅ **Monitoring** with Prometheus & Grafana
- ✅ **Testing** with pytest

---

## 📖 Need Help?

**👉 [Complete User Guide](COMPLETE_USER_GUIDE.md) has everything you need:**

- 🔧 Detailed setup instructions
- 🤖 Model training guide
- 🌐 API documentation
- 🐳 Docker deployment
- ⚙️ CI/CD pipeline setup
- 🧪 Testing strategies
- 📊 Monitoring setup
- 🛠️ Troubleshooting guide

---

**🎯 Happy MLOps! 🚀**
