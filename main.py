from flask import Flask, request, jsonify
import joblib
from confidenceIntervals import *
from sklearn.preprocessing import StandardScaler
import json


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
        final_score = estimate_elon_stats(value1, value2 , prediction)

        result = final_score[0].tolist(), final_score[1].tolist(), final_score[2]
       
            
        
        # Return the prediction
        return json.dumps({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})
    
def estimate_elon_stats(elo_rating_team_a, elo_rating_team_b , prediction):
    """
    Estimates the third-quarter stats for two teams based on their Elo ratings.
    Assumes a higher Elo rating indicates a stronger team likely to perform better.
    """
    
    strength_metric_team_a = elo_rating_team_a / prediction
    strength_metric_team_b = elo_rating_team_b / prediction

    score_benchmark = 12

    predicted_team = None
    
    away_3rdq = strength_metric_team_a * 100
    home_3rdq = strength_metric_team_b * 100

    if home_3rdq > score_benchmark:

        predicted_team = "HOME"
    else:
        predicted_team = "AWAY"
    
    return abs(home_3rdq), abs(away_3rdq), predicted_team

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
