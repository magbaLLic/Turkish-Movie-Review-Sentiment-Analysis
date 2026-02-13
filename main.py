import pandas as pd

df = pd.read_csv("C:\\Users\\ceren\\Desktop\\tr_nlp\\Turkish Movie Sentiment\\turkish_movie_sentiment_dataset.csv")
print("First 5 rows:")
print(df.head())

print("\nShape:", df.shape)
print("\nColumns:", df.columns)
print("\nPoint counts:")
print(df["point"].value_counts())
print("\nNull values:")
print(df.isnull().sum())

print("\nPoint counts (normalized):")
print(df["point"].value_counts(normalize=True))
print("\nUnique points:", sorted(df["point"].unique()))

def convert_label(rating):
    rating = float(str(rating).replace(',', '.'))
    if rating <= 2:
        return 0
    elif rating >= 4:
        return 1
    else:
        return None

df["label"] = df["point"].apply(convert_label)

df = df.dropna(subset=["label"])
print("\nLabel counts after conversion:")
print(df["label"].value_counts())

import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\r", " ", text)
    text = re.sub(r"\d+", "", text)  # SAYILARI SİL
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["clean_text"] = df["comment"].apply(clean_text)
print(df[["comment", "clean_text"]].head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]   # IMPORTANT (imbalance var çünkü)
)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,1)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Yeni vectorizer
vectorizer_balanced = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1,2)
)

X_train_tfidf_bal = vectorizer_balanced.fit_transform(X_train)
X_test_tfidf_bal = vectorizer_balanced.transform(X_test)

# Yeni model
model_balanced = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# Train
model_balanced.fit(X_train_tfidf_bal, y_train)

# Predict
y_pred_bal = model_balanced.predict(X_test_tfidf_bal)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred_bal))
print(classification_report(y_test, y_pred_bal))


feature_names = vectorizer_balanced.get_feature_names_out()
coefficients = model_balanced.coef_[0]

# Pozitif en güçlü 20 kelime
top_positive = sorted(zip(coefficients, feature_names), reverse=True)[:20]

# Negatif en güçlü 20 kelime
top_negative = sorted(zip(coefficients, feature_names))[:20]

print("Top Positive Words:")
for coef, word in top_positive:
    print(word, coef)

print("\nTop Negative Words:")
for coef, word in top_negative:
    print(word, coef)

