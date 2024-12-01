from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('finance.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    income = float(request.form['income'])
    fixed_expenses = float(request.form['fixed_expenses'])
    discretionary_expenses = float(request.form['discretionary_expenses'])
    risk_tolerance = int(request.form['risk_tolerance'])

    # Prepare the features for prediction
    features = pd.DataFrame([[income, fixed_expenses, discretionary_expenses]],
                            columns=['Income', 'Fixed Expenses', 'Discretionary Expenses'])

    # Predict savings using the model
    savings_prediction = model.predict(features)[0]

    # Generate investment recommendation based on risk tolerance
    if risk_tolerance == 1:
        investment_recommendation = "Low-risk investments: Bonds, Savings Accounts"
    elif risk_tolerance == 2:
        investment_recommendation = "Moderate-risk investments: Balanced ETFs, Mutual Funds"
    else:
        investment_recommendation = "High-risk investments: Stocks, Real Estate"

    # Render the result in an HTML template
    return render_template('result.html', 
                           savings=savings_prediction, 
                           investment_recommendation=investment_recommendation)

if __name__ == '__main__':
    app.run(debug=True)
