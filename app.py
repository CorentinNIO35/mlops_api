import joblib
import flask
import pandas as pd
from unidecode import unidecode

app = flask.Flask(__name__)

model = joblib.load('modelRF.v0.pickle')

def encoder(names):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    features = pd.DataFrame()
    for letter in alphabet:
        features[letter] = (
            names.apply(unidecode).str.upper().str.count(letter).astype(int)
        )
        
    return features


@app.route('/')
def hello():
    return f'Hello world'

@app.route('/predict/<prenom>',methods = ['GET'])
def predict(prenom):
    #test = pd.Series([prenom])
    #encode = encoder(pd.Series([prenom]))
    results = model.predict(encoder(pd.Series([prenom])))
    prediction = results[0]
    return "Female" if prediction == 1 else "Male"
    #return(prenom)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80)