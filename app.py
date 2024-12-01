from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np


app = Flask(__name__)


model = joblib.load('finance.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
       
        income = float(request.form['income'])
        fixed_expenses = float(request.form['fixed_expenses'])
        discretionary_expenses = float(request.form['discretionary_expenses'])
        risk_tolerance = request.form['risk_tolerance']

       
        X = pd.DataFrame([[income, fixed_expenses, discretionary_expenses]], 
                         columns=['Income', 'Fixed Expenses', 'Discretionary Expenses'])

       
        predicted_savings = model.predict(X)[0]

       
        fixed_expense_ratio = 0.50
        discretionary_ratio = 0.30
        savings_ratio = 0.20

        suggested_fixed_expenses = income * fixed_expense_ratio
        suggested_discretionary_expenses = income * discretionary_ratio
        suggested_savings = income * savings_ratio

       
        if risk_tolerance == '1':
            investment_recommendation = "Low-risk investments: Bonds, Savings Accounts"
        elif risk_tolerance == '2':
            investment_recommendation = "Moderate-risk investments: Balanced ETFs, Mutual Funds"
        elif risk_tolerance == '3':
            investment_recommendation = "High-risk investments: Stocks, Real Estate"
        else:
            investment_recommendation = "Invalid risk tolerance value provided."

        
        return jsonify({
            'Predicted Savings': f"${predicted_savings:.2f}",
            'Suggested Fixed Expenses': f"${suggested_fixed_expenses:.2f}",
            'Suggested Discretionary Expenses': f"${suggested_discretionary_expenses:.2f}",
            'Suggested Savings': f"${suggested_savings:.2f}",
            'Investment Recommendation': investment_recommendation
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
