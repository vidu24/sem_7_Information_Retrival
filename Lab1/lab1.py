import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Input dataset
data = [
    ("I love this product. It is excellent and works perfectly.", "Positive"),
    ("The movie was amazing and I really enjoyed it.", "Positive"),
    ("The service was fast, friendly and wonderful.", "Positive"),
    ("This is a great experience and I am very happy.", "Positive"),
    ("The food was delicious and the staff were excellent.", "Positive"),
    ("I hate this product. It is terrible and completely useless.", "Negative"),
    ("The movie was boring and extremely disappointing.", "Negative"),
    ("The service was slow and the staff were rude.", "Negative"),
    ("This was a horrible experience and I am very unhappy.", "Negative"),
    ("The food was awful and the restaurant was disappointing.", "Negative")
]

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

actual = []
predicted = []

for i, (text, label) in enumerate(data, 1):

    print("\n" + "=" * 60)
    print("TEXT", i)
    print("=" * 60)

    print("\nOriginal Text:")
    print(text)

    # Text cleaning and normalization
    clean = re.sub(r'[^a-zA-Z\s]', '', text).lower()

    print("\n1. Cleaned and Normalized:")
    print(clean)

    # Tokenization
    tokens = word_tokenize(clean)

    print("\n2. Tokenized:")
    print(tokens)

    # Stop word removal
    filtered = [w for w in tokens if w not in stop_words]

    print("\n3. Stop Words Removed:")
    print(filtered)

    # Stemming
    stemmed = [stemmer.stem(w) for w in filtered]

    print("\n4. Stemmed:")
    print(stemmed)

    # Lemmatization
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]

    print("\n5. Lemmatized:")
    print(lemmatized)

    # Sentiment classification
    score = sia.polarity_scores(text)['compound']
    pred = "Positive" if score >= 0 else "Negative"

    print("\n6. Classification:")
    print("Actual Label:", label)
    print("Predicted Label:", pred)

    actual.append(1 if label == "Positive" else 0)
    predicted.append(1 if pred == "Positive" else 0)

# Evaluation
cm = confusion_matrix(actual, predicted)
precision = precision_score(actual, predicted, zero_division=0)
recall = recall_score(actual, predicted, zero_division=0)
f1 = f1_score(actual, predicted, zero_division=0)

print("\n" + "=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

print("\nConfusion Matrix:")
print(cm)

print("\nPrecision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))

# Confusion matrix visualization
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()