from flask import Flask,request, url_for, redirect, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

model=pickle.load(open('final_model.pkl','rb'))
def make_prediction(inputs):
    """Run model prediction given inputs list"""
    features = np.array([inputs])  # wrap into 2D array for sklearn
    return model.predict(features)[0]   # get first prediction


@app.route('/')
def home():
    return render_template('index.html', risk=None)

@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    try:
        # Strings (keep as-is, don't cast to int)
        Age_Category = request.form['age']               # e.g. "18-24"
        General_Health = request.form['general_health']  # e.g. "Good"
        Checkup = request.form['checkup']                # e.g. "Within the past year"
        Smoking_History = request.form['smoking_history']  # "Yes"/"No"
        Arthritis = request.form['arthritis']
        Diabetes = request.form['diabetes']
        Depression = request.form['depression']
        Other_Cancer = request.form['other_cancer']
        Skin_Cancer = request.form['skin_cancer']
        Sex = request.form['sex']                        # "Male"/"Female"
        Exercise = request.form['exercise']              # "Yes"/"No"

        # Numeric fields (cast to float)
        Height_cm = float(request.form['height_cm'])
        Weight_kg = float(request.form['weight_kg'])
        BMI = float(request.form['bmi'])
        Alcohol_Consumption = float(request.form['alcohol_consumption'])
        Fruit_Consumption = float(request.form['fruit_consumption'])
        Green_Vegetables_Consumption = float(request.form['green_vegetables_consumption'])
        FriedPotato_Consumption = float(request.form['fried_potato_consumption'])

    except Exception as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

    # ✅ Build DataFrame with correct column names and expected string categories
    features = pd.DataFrame([{
        'Arthritis': Arthritis,
        'Depression': Depression,
        'Diabetes': Diabetes,
        'Exercise': Exercise,
        'Other_Cancer': Other_Cancer,
        'Sex': Sex,
        'Skin_Cancer': Skin_Cancer,
        'Smoking_History': Smoking_History,
        'Age_Category': Age_Category,
        'Checkup': Checkup,
        'General_Health': General_Health,
        'Height_(cm)': Height_cm,
        'Weight_(kg)': Weight_kg,
        'BMI': BMI,
        'Alcohol_Consumption': Alcohol_Consumption,
        'Fruit_Consumption': Fruit_Consumption,
        'Green_Vegetables_Consumption': Green_Vegetables_Consumption,
        'FriedPotato_Consumption': FriedPotato_Consumption
    }])

    # Run prediction
    risk_prob = model.predict_proba(features)[0][1] * 100

    # Use your mapping function from model.py
    from model import risk_category
    map_risk = risk_category(risk_prob)   # adjust if your function expects 0–1

    return render_template('risk.html', risk=risk_prob, category=map_risk)



if __name__ == '__main__':
    app.run(debug=True)