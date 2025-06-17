from flask import Flask, request, render_template
import os
import pickle
import pandas as pd
from assets_data_prep import prepare_data

app = Flask(__name__)

# Load the full pipeline model (including preprocessor)
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Binary fields from checkboxes - default to 0 if not checked
    binary_fields = ['has_parking', 'has_storage', 'elevator', 'ac', 'handicap',
                     'has_safe_room', 'has_balcony', 'is_renovated', 'is_furnished']

    # Collect fields from form
    features = {
        'property_type': request.form.get('property_type', ''),
        'neighborhood': request.form.get('neighborhood', ''),
        'floor': int(request.form.get('floor', 0)),
        'area': int(request.form.get('area', 0)),
        'room_num': int(request.form.get('room_num', 0)),
        'total_floors': int(request.form.get('total_floors', 0)),
        'building_tax': float(request.form.get('building_tax', 0)),
        'monthly_arnona': float(request.form.get('monthly_arnona', 0)),
        'distance_group': float(request.form.get('distance_group', 1)),
        'garden_area': int(request.form.get('garden_area', 0))
    }

    # Add binary features
    for field in binary_fields:
        features[field] = int(request.form.get(field, 0))

    # Derived feature
    features['rooms_per_sqm'] = features['room_num'] / features['area'] if features['area'] else 0
    features.pop('room_num')

    feature_names = ['property_type', 'neighborhood', 'floor', 'area', 'has_parking', 'has_storage', 'elevator',
                     'ac', 'handicap', 'has_safe_room', 'total_floors', 'building_tax', 'has_balcony',
                     'is_renovated', 'is_furnished', 'rooms_per_sqm', 'distance_group', 'monthly_arnona', 'garden_area']

    input_data = pd.DataFrame([features], columns=feature_names)
    input_data = prepare_data(input_data, 'test')

    pipeline = model.best_estimator_
    predicted_price = pipeline.predict(input_data)[0]

    return render_template('index.html', prediction_text='Predicted Price: {:.2f}'.format(predicted_price))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
