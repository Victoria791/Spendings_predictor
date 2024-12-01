from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd


model = joblib.load('finance.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   
    income = float(request.form['income'])
    fixed_expenses = float(request.form['fixed_expenses'])
    discretionary_expenses = float(request.form['discretionary_expenses'])
    risk_tolerance = int(request.form['risk_tolerance'])

    
    features = pd.DataFrame([[income, fixed_expenses, discretionary_expenses]],
                            columns=['Income', 'Fixed Expenses', 'Discretionary Expenses'])


    savings_prediction = model.predict(features)[0]

   
    if risk_tolerance == 1:
        investment_recommendation = "Low-risk investments: Bonds, Savings Accounts"
    elif risk_tolerance == 2:
        investment_recommendation = "Moderate-risk investments: Balanced ETFs, Mutual Funds"
    else:
        investment_recommendation = "High-risk investments: Stocks, Real Estate"

    
    if request.accept_mimetypes.best == 'application/json':
       
        return jsonify({
            'savings': savings_prediction,
            'investment_recommendation': investment_recommendation
        })
    else:
    
        return render_template('result.html',
                               savings=savings_prediction,
                               investment_recommendation=investment_recommendation)

if __name__ == '__main__':
    app.run(debug=True)
