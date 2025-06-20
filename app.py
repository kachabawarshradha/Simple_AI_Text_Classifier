
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Sample AI logic: classify text as "tech" or "non-tech"
train_data = ["python code", "machine learning", "data analysis", "football", "cooking", "travel"]
train_labels = ["tech", "tech", "tech", "non-tech", "non-tech", "non-tech"]
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
model = MultinomialNB()
model.fit(X_train, train_labels)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_input = request.form["text"]
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)
        result = prediction[0]
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
