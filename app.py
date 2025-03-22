from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Route for the home page
@app.route('/')
def home():
    return render_template("index.html", prediction=None)

# Route for processing the review
@app.route('/analyze', methods=['POST'])
def analyze():
    review_text = request.form.get('review')  # Get text input from form
    if not review_text:
        return render_template('index.html', prediction="Please enter a review.")

    # Vectorize input review
    vectorized_review = vectorizer.transform([review_text])

    # Make prediction
    prediction = model.predict(vectorized_review)[0]

    # Map prediction to sentiment label
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    # Debugging outputs
    print("User Input:", review_text)  # Print raw user input
    print("Vectorized Input:", vectorized_review.toarray())  # Print numeric transformation
    print("Prediction:", prediction)  # Print prediction result
    print("Predicted Sentiment:", sentiment_map[prediction])  # Print final sentiment label

    return render_template('index.html', prediction=sentiment_map[prediction])

if __name__ == "__main__":
    app.run(debug=True)
