import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
data = pd.read_csv('../data/weekly_demand.csv')

# Features (X) and target (y)
X = data[['price', 'ads_spent']]
y = data['sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Ensure model directory exists
os.makedirs('../model', exist_ok=True)

# Save model
joblib.dump(model, '../model/demand_model.pkl')

print("✅ Model trained and saved as demand_model.pkl")