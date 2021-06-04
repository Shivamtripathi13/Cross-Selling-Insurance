import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('fmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if(output==0):
        return render_template('index.html', prediction_text='Yes, the customer is interested to get Vehicle Insurance')
    else:
        return render_template('index.html', prediction_text='No, the customer is not interested to get Vehicle Insurance')


if __name__ == "__main__":
    app.run(debug=True)