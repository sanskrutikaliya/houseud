
import pandas as pd
from flask import Flask, app,render_template,request

import pickle 
import numpy as np 

pipe = pd.read_pickle("RidgeModel.pkl")

app= Flask(__name__,template_folder = 'template') 
data=pd.read_csv('Cleaned_data.csv')
print(data)
pipe=pickle.load(open('RidgeModel.pkl','rb'))

@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html', locations = locations)

@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    sqft=request.form.get('total_sqft')
    print(location,bhk,bath,sqft)
    input=pd.DataFrame([[location,sqft,bath,bhk]], columns=['location','total_sqft','bath','bhk'])
    pipe.predict(input)
    prediction=pipe.predict(input)[0]*1e5

    return str(np.round(prediction,2))



if __name__=="__main__":
    app.run(debug=True,port=5500)
