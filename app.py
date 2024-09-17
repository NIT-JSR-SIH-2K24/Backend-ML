from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Train a simple Logistic Regression model for demo purposes
# You can replace this with loading a pre-trained model
model = LogisticRegression()
X = pd.DataFrame({
    'feature1': [1, 2, 3, 4],
    'feature2': [10, 20, 30, 40]
})
y = [0, 1, 0, 1]
model.fit(X, y)

@app.route('/')
def home():
    return "ML Model Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['feature1'], data['feature2']]).reshape(1, -1)
    
    prediction = model.predict(features)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
