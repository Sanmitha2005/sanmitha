from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

# Load dataset and model
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def fit(X, y, epochs=100, learning_rate=0.01):
    w = np.zeros(X.shape[1])
    cost_list = []
    m = len(y)
    for epoch in range(epochs):
        y_pred = sigmoid(np.dot(X, w))
        error = y_pred - y
        grad = np.dot(X.T, error) / m
        w -= learning_rate * grad
        cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        cost_list.append(cost)
    return w, cost_list

# Load dataset and prepare model
filepath = 'instagram_accounts.csv'
df = load_dataset(filepath)

X = df.drop(columns=['Fake'])
y = df['Fake']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

epochs = 1000
learning_rate = 0.01
w, _ = fit(X_scaled, y, epochs, learning_rate)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    followers = float(request.form['followers'])
    following = float(request.form['following'])
    posts = float(request.form['posts'])
    likes = float(request.form['likes'])
    comments = float(request.form['comments'])
    bio_length = float(request.form['bio_length'])

    input_data = np.array([followers, following, posts, likes, comments, bio_length])
    input_data = scaler.transform([input_data])
    input_data = np.hstack([np.ones((input_data.shape[0], 1)), input_data])

    prob = sigmoid(np.dot(input_data, w))
    prediction = np.where(prob > 0.5, 1, 0)

    return render_template('result.html', probability=prob[0], prediction='Fake' if prediction[0] == 1 else 'Real')

if __name__ == '__main__':
    app.run(debug=True)
