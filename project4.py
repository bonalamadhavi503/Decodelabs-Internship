# 1. IMPORT LIBRARIES
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 2. DOWNLOAD NLTK DATA

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 3. DATASET 
df = pd.DataFrame({
    "Review": [
        "I love this product",
        "Amazing quality and very useful",
        "Excellent service and product",
        "Best purchase ever",
        "Very good item",
        "I am very happy",
        "Fantastic experience",
        "Highly recommended product",

        "Worst product ever",
        "I hate this item",
        "Very bad experience",
        "Not worth the money",
        "Totally disappointed",
        "Poor quality product",
        "I will never buy this again",
        "Waste of money"
    ],
    "Sentiment": [
        "Positive","Positive","Positive","Positive",
        "Positive","Positive","Positive","Positive",
        "Negative","Negative","Negative","Negative",
        "Negative","Negative","Negative","Negative"
    ]
})

# 4. TEXT PREPROCESSING FUNCTION
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

df["Clean_Review"] = df["Review"].apply(preprocess)


# 5. LABEL ENCODING
df["Sentiment"] = df["Sentiment"].map({"Positive": 1, "Negative": 0})

# 6. TF-IDF VECTORIZATION

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Clean_Review"])
y = df["Sentiment"]

# 7. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 8. MODEL (NAIVE BAYES - REQUIRED)
model = MultinomialNB()
model.fit(X_train, y_train)

# 9. PREDICTION
y_pred = model.predict(X_test)

# 10. RESULTS
print("\nACCURACY:", accuracy_score(y_test, y_pred))
print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))

 
# 11. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 12. TEST FUNCTION
def predict_sentiment(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    clean = " ".join(tokens)

    vec = vectorizer.transform([clean])
    pred = model.predict(vec)

    return "Positive" if pred[0] == 1 else "Negative"

# 13. TEST EXAMPLES
print(predict_sentiment("This product is amazing"))
print(predict_sentiment("Worst experience ever"))