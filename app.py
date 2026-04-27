import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset (you can replace later)
data = {
    "text": [
        "Breaking news: something big happened",
        "Click here to win money!!!",
        "Government announces new policy",
        "This is fake and misleading news"
    ],
    "label": [1, 0, 1, 0]  # 1 = real, 0 = fake
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Test prediction
def predict_news(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "Real News" if prediction[0] == 1 else "Fake News"

print(predict_news("Win a free iPhone now!!!"))
