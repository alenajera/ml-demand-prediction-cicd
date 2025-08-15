from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

# Basic logging setup
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# ✅ Ruta corregida para Docker y local
model = joblib.load('model/demand_model.pkl')
logging.info("✅ Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        price = data.get('price')
        ads_spent = data.get('ads_spent')

        if price is None or ads_spent is None:
            return jsonify({'error': 'Please provide price and ads_spent'}), 400

        prediction = model.predict(np.array([[price, ads_spent]]))
        logging.info(f"Prediction made for price={price}, ads_spent={ads_spent}")
        return jsonify({'predicted_sales': float(prediction[0])})

    except Exception as e:
        logging.error(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Escucha en 0.0.0.0:5000 para que Docker pueda exponerlo
    app.run(host='0.0.0.0', port=5000, debug=True)