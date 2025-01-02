from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

# Use raw strings to avoid invalid escape sequences
data_path = r"D:\Project Prediction\Cleaned_data.csv"
model_path = r"D:\Project Prediction\RidgeModel.pkl"

data = pd.read_csv(data_path)
pipe = pickle.load(open(model_path, "rb"))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    
    print(location, bhk, bath, sqft)
    
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0] * 1e5
    
    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
