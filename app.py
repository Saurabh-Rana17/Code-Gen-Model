from flask import Flask, request, jsonify 
from flask_cors import CORS

import joblib

# Create a Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('model.pkl')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get keyword from request
    keyword = request.json.get('keyword')

    # Make prediction
    predicted_code = model.predict([keyword])

    # Return prediction
    return jsonify({'predicted_code': predicted_code[0]})

if __name__ == '__main__':
    app.run(debug=True)
