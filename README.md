# Turkish-Movie-Review-Sentiment-Analysis
This project implements a sentiment classification pipeline for Turkish movie reviews using classical NLP methods.

Dataset

83k movie reviews

Rating-based sentiment labels

Neutral ratings removed

Preprocessing

Lowercasing

Digit removal (to prevent rating leakage)

Punctuation removal

Whitespace normalization

Models

Baseline Logistic Regression (unigram)

Balanced Logistic Regression (class_weight="balanced")

Balanced + Bigram TF-IDF

Final Model Performance
Metric	Value
Accuracy	0.872
Negative Recall	0.83
Macro F1	0.84
Key Insight

Numeric rating expressions (e.g., “10/10”) initially influenced the classifier. After removing digits, the model relied purely on lexical sentiment features.

Example Learned Features

Positive:

harika

mükemmel

en iyi

Negative:

berbat

rezalet

vasat


This project explores sentiment classification of Turkish movie reviews using TF-IDF features and logistic regression. After addressing class imbalance and removing numeric rating leakage, the model achieved 87.2% accuracy and significantly improved minority-class recall. Feature analysis confirmed that the classifier relies on evaluative lexical cues rather than explicit rating indicators.

Data: kaggle.com/datasets/mustfkeskin/turkish-movie-sentiment-analysis-dataset
