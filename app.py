from flask import Flask, render_template, request
import pickle
import numpy as np
from scipy.spatial import distance  # Ensure this import is included

# Define the SimpleRecommender class
class SimpleRecommender:
    def __init__(self, average_features):
        self.average_features = average_features

    def recommend(self, features):
        distances = self.average_features.apply(lambda x: distance.euclidean(x, features), axis=1)
        recommended_code = distances.idxmin()
        return recommended_code

# Load the recommender model
with open('recommender_model.pkl', 'rb') as file:
    recommender = pickle.load(file)

# Load fertilizer maps
with open('fertilizer_map.pkl', 'rb') as file:
    fertilizer_map = pickle.load(file)

with open('reverse_fertilizer_map.pkl', 'rb') as file:
    reverse_fertilizer_map = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve and convert input data
        Nitrogen = request.form.get('Nitrogen', '0')
        Potassium = request.form.get('Potassium', '0')
        Phosphorous = request.form.get('Phosphorous', '0')

        # Ensure the data is numerical
        try:
            Nitrogen = float(Nitrogen)
            Potassium = float(Potassium)
            Phosphorous = float(Phosphorous)
        except ValueError:
            return render_template('index.html', result="Error: Please enter valid numerical values.")

        # Create input array for prediction
        input_data = np.array([Nitrogen, Potassium, Phosphorous])

        # Predict
        result_code = recommender.recommend(input_data)

        # Map result code to the corresponding fertilizer name
        fertilizer_name = reverse_fertilizer_map.get(result_code, 'Unknown Fertilizer')

    except Exception as e:
        result = f"Error: {str(e)}"
        return render_template('index.html', result=result)

    return render_template('index.html', result=fertilizer_name)

if __name__ == '__main__':
    app.run(debug=True)
