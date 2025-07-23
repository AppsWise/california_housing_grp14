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


file_path = 'artifacts/housing.csv'
df = pd.read_csv(file_path)

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

# Train both
linear_pipeline.fit(X_train, y_train)
tree_pipeline.fit(X_train, y_train)

# Step 4: Evaluate Both Models
def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    print(f"\n{name} Performance:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
    print(f"R² Score: {r2_score(y_test, preds):.4f}")

evaluate_model("Linear Regression", linear_pipeline, X_test, y_test)
evaluate_model("Decision Tree Regressor", tree_pipeline, X_test, y_test)


