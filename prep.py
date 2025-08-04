import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import pickle
import os


file_path = 'artifacts/housing.csv'
df = pd.read_csv(file_path)

# Set up MLflow
mlflow.set_experiment("california_housing_experiment")

# Step 1: Feature Engineering
# Fill missing first
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# Feature engineering
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']


# Step 2: Split X and y
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Identify column types
numeric_features = X.select_dtypes(include=['float64', 'int']).columns.tolist()
categorical_features = ['ocean_proximity']

# Pipeline 1: Linear Regression
# Apply log1p to skewed numeric columns
skewed_cols = ['total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']

numeric_transformer_linear = Pipeline(steps=[
    ('log_transform', FunctionTransformer(func=np.log1p, feature_names_out='one-to-one')),
    ('scaler', StandardScaler())
])

# Only log-transform skewed columns, rest pass through
preprocessor_linear = ColumnTransformer(
    transformers=[
        ('num_log', numeric_transformer_linear, skewed_cols),
        ('num_pass', StandardScaler(), list(set(numeric_features) - set(skewed_cols))),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# Linear Regression pipeline
linear_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_linear),
    ('regressor', LinearRegression())
])

# Pipeline 2: Decision Tree Regressor
# Decision Trees don’t need log or scaling
preprocessor_tree = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_tree),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

# Step 3: Train-Test Split and Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and track model with MLflow
def train_and_track_model(model_name, pipeline, model_params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        
        # Log model
        mlflow.sklearn.log_model(pipeline, "model")
        
        print(f"\n{model_name} Performance:")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Train R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        
        return pipeline, test_rmse, test_r2

# Train Linear Regression
linear_params = {
    "model_type": "LinearRegression",
    "preprocessing": "log_transform_and_scaling",
    "skewed_columns": "total_rooms,total_bedrooms,population,households,median_income"
}

linear_model, linear_rmse, linear_r2 = train_and_track_model(
    "Linear_Regression", 
    linear_pipeline, 
    linear_params, 
    X_train, X_test, y_train, y_test
)

# Train Decision Tree
tree_params = {
    "model_type": "DecisionTreeRegressor",
    "preprocessing": "minimal_preprocessing",
    "random_state": 42
}

tree_model, tree_rmse, tree_r2 = train_and_track_model(
    "Decision_Tree", 
    tree_pipeline, 
    tree_params, 
    X_train, X_test, y_train, y_test
)

# Select best model based on test R² score
if tree_r2 > linear_r2:
    best_model = tree_model
    best_model_name = "Decision_Tree"
    best_rmse = tree_rmse
    best_r2 = tree_r2
else:
    best_model = linear_model
    best_model_name = "Linear_Regression"
    best_rmse = linear_rmse
    best_r2 = linear_r2

print(f"\nBest Model: {best_model_name}")
print(f"Best Test RMSE: {best_rmse:.2f}")
print(f"Best Test R² Score: {best_r2:.4f}")

# Register the best model in MLflow Model Registry
with mlflow.start_run(run_name=f"Best_Model_{best_model_name}"):
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_test_rmse", best_rmse)
    mlflow.log_metric("best_test_r2", best_r2)
    
    # Log and register the best model
    model_uri = mlflow.sklearn.log_model(
        best_model, 
        "best_model",
        registered_model_name="california_housing_best_model"
    )
    
    print(f"Best model registered in MLflow Model Registry with URI: {model_uri}")

# Save the best model as pickle for Flask app compatibility
os.makedirs('artifacts', exist_ok=True)
with open('artifacts/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Best model saved as artifacts/model.pkl for Flask app")


