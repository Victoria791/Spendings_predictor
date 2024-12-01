from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('finance.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user inputs
        income = float(request.form['income'])
        fixed_expenses = float(request.form['fixed_expenses'])
        discretionary_expenses = float(request.form['discretionary_expenses'])
        risk_tolerance = request.form['risk_tolerance']

        # Create a DataFrame for predictions
        X = pd.DataFrame([[income, fixed_expenses, discretionary_expenses]], 
                         columns=['Income', 'Fixed Expenses', 'Discretionary Expenses'])

        # Predict savings
        predicted_savings = model.predict(X)[0]

        # Suggest budget allocations
        fixed_expense_ratio = 0.50
        discretionary_ratio = 0.30
        savings_ratio = 0.20

        suggested_fixed_expenses = income * fixed_expense_ratio
        suggested_discretionary_expenses = income * discretionary_ratio
        suggested_savings = income * savings_ratio

        # Determine investment recommendation
        if risk_tolerance == '1':
            investment_recommendation = "Low-risk investments: Bonds, Savings Accounts"
        elif risk_tolerance == '2':
            investment_recommendation = "Moderate-risk investments: Balanced ETFs, Mutual Funds"
        elif risk_tolerance == '3':
            investment_recommendation = "High-risk investments: Stocks, Real Estate"
        else:
            investment_recommendation = "Invalid risk tolerance value provided."

        # Return predictions and recommendations
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
