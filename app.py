from flask import Flask, request, jsonify
import joblib
from preprocessing import lemmatize_text

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define a default route
@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Sentiment Analysis API!"

# Define a route for sentiment prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Assumes JSON input with a 'text' field
    text = data.get("text", "")
    if text:
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        sentiment = "positive" if prediction == 1 else "negative"
        return jsonify({"text": text, "sentiment": sentiment})
    else:
        return jsonify({"error": "No text provided"}), 400

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

