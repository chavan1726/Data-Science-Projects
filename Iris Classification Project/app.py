from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import joblib


app = Flask(__name__)

model=joblib.load("model.pkl")
scaler=joblib.load("scaler.pkl")
encoder=joblib.load("encoder.pkl")



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    Input={ 'sepal_length':[sepal_length], 
       'sepal_width':[sepal_width], 
       'petal_length':[petal_length], 
       'petal_width':[petal_width]
    }
    Input=pd.DataFrame(Input)

    standard=scaler.transform(Input)
    x=pd.DataFrame(standard,columns=Input.columns)

   
    prediction = model.predict(x)
    prediction=encoder.inverse_transform(prediction)
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
