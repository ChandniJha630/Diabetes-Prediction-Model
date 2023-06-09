from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application
#import ridge and scaler model
model=pickle.load(open('models/Diabetes_prediction.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))
#Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
@app.route('/',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        Pregnancies=float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

  
        new_data_scaled=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        result=model.predict(new_data_scaled)
        if result==0:
            ans="You are not having Diabetes"
        else:
            ans="You are having Diabetes"
        return render_template('home.html',result=ans)

    else:
        return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")
