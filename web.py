from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Simple training data
texts = [
    "Breaking news something important happened",
    "Win money now click here",
    "Government policy update",
    "Fake misleading news spread online"
]
labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
</head>
<body>
    <h2>Fake News Detector</h2>
    <form method="post">
        <input type="text" name="news" placeholder="Enter news text" size="50">
        <button type="submit">Check</button>
    </form>
    <h3>{{ result }}</h3>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    if request.method == 'POST':
        news = request.form['news']
        vec = vectorizer.transform([news])
        pred = model.predict(vec)
        result = "Real News" if pred[0] == 1 else "Fake News"
    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(debug=True)
