from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load the saved model and encoders
model = joblib.load('model/titanic_survival_model.pkl')
sex_encoder = joblib.load('model/sex_encoder.pkl')
embarked_encoder = joblib.load('model/embarked_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']
        
        # Encode categorical variables
        sex_encoded = sex_encoder.transform([sex])[0]
        embarked_encoded = embarked_encoder.transform([embarked])[0]
        
        # Create feature array in the correct order
        features = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex_encoded],
            'Age': [age],
            'Fare': [fare],
            'Embarked': [embarked_encoded]
        })
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Get result
        if prediction == 1:
            result = "✅ SURVIVED"
            message = "This passenger would likely have survived the Titanic disaster."
            survival_prob = probability[1] * 100
        else:
            result = "❌ DID NOT SURVIVE"
            message = "This passenger would likely not have survived the Titanic disaster."
            survival_prob = probability[0] * 100
        
        return render_template('index.html', 
                             prediction=result,
                             message=message,
                             probability=f"{survival_prob:.1f}%")
    
    except Exception as e:
        return render_template('index.html', 
                             prediction="Error",
                             message=f"An error occurred: {str(e)}",
                             probability="N/A")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)