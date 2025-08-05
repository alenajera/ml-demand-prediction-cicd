import joblib
import numpy as np

# Load the trained model
model = joblib.load('../model/demand_model.pkl')

# Example data: price=9.0, ads_spent=750
sample_data = np.array([[9.0, 750]])

# Make prediction
prediction = model.predict(sample_data)
print(f"Predicted weekly sales: {prediction[0]:.2f}")
