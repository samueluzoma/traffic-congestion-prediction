from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the saved assets
# We use os.path.join to ensure paths work across different operating systems
try:
    model = joblib.load('traffic_model.joblib')
    le_day = joblib.load('day_encoder.joblib')
    le_situation = joblib.load('situation_encoder.joblib')
    feature_names = joblib.load('feature_names.joblib')
    print("Model and encoders loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")

@app.route('/', methods=['GET'])
def home():
    return "Traffic Prediction API is running locally!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Convert to DataFrame
        df_input = pd.DataFrame([data])
        
        # 1. Encode the 'Day of the week' using the saved encoder
        # We handle potential unseen labels by wrapping in a try-except
        df_input['Day of the week'] = le_day.transform(df_input['Day of the week'])
        
        # 2. Ensure columns are in the correct order as per training
        df_input = df_input[feature_names]
        
        # 3. Make prediction
        prediction_id = model.predict(df_input)[0]
        
        # 4. Convert numeric prediction back to text (e.g., 0 -> 'heavy')
        prediction_label = le_situation.inverse_transform([prediction_id])[0]
        
        # 5. Get probability (optional, but helpful)
        probability = max(model.predict_proba(df_input)[0]) * 100
        
        return jsonify({
            'status': 'success',
            'prediction': prediction_label,
            'confidence': f"{probability:.2f}%"
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    # Run locally on port 5000
    app.run(debug=True, port=5004)