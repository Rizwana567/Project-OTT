from flask import Flask, request, render_template
import pickle
import numpy as np

# Load trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Features used in training
feature_columns = ["age", "no_of_days_subscribed", "weekly_mins_watched",
                   "minimum_daily_mins", "maximum_daily_mins", "weekly_max_night_mins"]

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = [float(request.form[col]) for col in feature_columns]

        # Convert input to numpy array and reshape
        features = np.array(data).reshape(1, -1)

        # Scale numerical features
        features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features)[0]
        result = "Churn" if prediction == 1 else "No Churn"

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
