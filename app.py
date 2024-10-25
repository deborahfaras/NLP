from flask import Flask, request, jsonify
from flask import Flask, render_template, request
import joblib
from preprocessing import lemmatize_text

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    input_text = request.form['review_text']
    
    # Transform input text using the vectorizer
    input_vector = vectorizer.transform([input_text])
    
    # Predict sentiment
    prediction = model.predict(input_vector)[0]
    
    # Convert prediction to positive/negative text
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return render_template('index.html', prediction=sentiment)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

