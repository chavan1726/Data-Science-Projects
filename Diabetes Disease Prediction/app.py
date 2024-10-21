from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as  pd

app = Flask(__name__)

# Load the model and scaler
model=joblib.load("model.pkl")
scaler=joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    data = {
        'Pregnancies': request.form['Pregnancies'],
        'Glucose': request.form['Glucose'],
        'BloodPressure': request.form['BloodPressure'],
        'SkinThickness': request.form['SkinThickness'],
        'Insulin': request.form['Insulin'],
        'BMI': request.form['BMI'],
        'DiabetesPedigreeFunction': request.form['DiabetesPedigreeFunction'],
        'Age': request.form['Age']
    }

    # Prepare the input for the model
    input_data = pd.DataFrame(data)
    # Scale the input
    scaled_input = scaler.transform(input_data)
    input=pd.DataFrame(scaled_input,columns=input_data.columns)

    # Make prediction
    prediction = model.predict(input)
    result = 'Positive' if prediction[0] == 1 else 'Negative'

    return render_template('result.html', input_data=data, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
