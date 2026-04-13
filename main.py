import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data
df = pd.read_csv('data/solarglass_tweets.csv')

# 2. Simple Preprocessing (Lowercasing and cleaning)
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

df['cleaned_text'] = df['Tweet_Text'].apply(clean_text)

# 3. Data Splitting (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['Label'], test_size=0.20, random_state=42
)

# 4. Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Classifier 1: Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_preds = nb_model.predict(X_test_tfidf)

# 6. Classifier 2: Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)

# 7. Results
print("--- Naive Bayes Report ---")
print(classification_report(y_test, nb_preds))

print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, lr_preds))
