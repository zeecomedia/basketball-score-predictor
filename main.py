from flask import Flask, request, jsonify
import joblib
from confidenceIntervals import *
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Load the model
loaded_model = joblib.load('./basketball_score_prediction_model.pkl')

@app.route('/', methods=['GET'])
def index():
    return 'Working'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from POST request
        data = request.get_json(force=True)
        
        # Extract features from the data
        # Assuming the model expects two features (replace 'feature1' and 'feature2' with actual feature names)
        value1 = data['feature1']
        value2 = data['feature2']
        
        scaler = StandardScaler()
        # Prepare the data for prediction
        new_data = [[value1, value2]]
        scaled_new_data = scaler.fit_transform(new_data)  
        new_data = scaler.transform(scaled_new_data)  

        
        # Make prediction  
        prediction = loaded_model.predict(new_data)
        
        # Return the prediction
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
