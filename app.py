
from ctypes.wintypes import MSG
from pyexpat import model
from flask import Flask, request, render_template
import numpy as np
import pandas
import sklearn
import pickle

# Create the Flask app instance here
app = Flask(__name__)

# ... rest of your code to load models, define routes, etc.


@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Correct scaling: only use the MinMaxScaler
        scaled_features = MSG.transform(single_pred)
        prediction = model.predict(scaled_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
        
        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = "{} is a best crop to be cultivated".format(crop)
        else:
            result = "Sorry, we are not able to recommend a proper crop for this environment"
    
    return render_template('index.html', result=result)