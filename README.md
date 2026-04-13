#  Assignment Title

## (1) Problem Statement
Social media platforms like Twitter (X) generate vast amounts of unstructured data. For a tech company launching a new product like the SolarGlass Smartphone, it is nearly impossible to manually track every user's opinion. This project addresses the need for an automated system to categorize public sentiment to understand market reception.

## (2) Objective
- To collect and manually label a dataset of 100 tweets regarding "SolarGlass" technology.
- To implement and compare machine learning classifiers (Naive Bayes and Logistic Regression).
- To evaluate models using Precision and Recall to determine the most effective approach for sentiment detection.

## (3) Dataset
- Source: Manually curated dataset of 100 tweets related to the "SolarGlass Smartphone" (2026 Buzzword).
- Features: Tweet_Text (Raw text), Label (Categorical: 1 for Positive, 0 for Neutral, -1 for Negative), cleaned_text (Processed text).
- Size: 100 rows (80 for training, 20 for testing).

## (4) Methodology
1. Data Preprocessing: Converted text to lowercase, removed punctuation, URLs, and stopwords to reduce noise.
2. EDA: Analyzed the distribution of sentiments (Positive vs. Negative vs. Neutral).
3. Model Building: Utilized TfidfVectorizer for feature extraction and trained Multinomial Naive Bayes and Logistic Regression models.
4. Evaluation: Generated a classification report to measure performance on the 20-tweet test set.

## (5) Results
- Metrics and insights#----
- Better Performer: Logistic Regression outperformed Naive Bayes in this specific task. It showed superior balance between precision and recall across all classes, specifically achieving an F1-score of 0.86 for positive sentiments.

- Class -1 Performance: Both models were excellent at identifying negative tweets (100% recall), likely due to specific "red flag" keywords (e.g., "waste," "shattered," "scam") present in the dataset.

- Improvements: Accuracy could be further improved by expanding the dataset beyond 100 tweets and using advanced techniques like Lemmatization or using pre-trained word embeddings (Word2Vec).

## (6) How to Run
```bash
pip install -r requirements.txt
python main.py
```

## (7) Conclusion
The project successfully demonstrates that even with a small dataset of 100 tweets, machine learning models can effectively categorize sentiment. While Naive Bayes is faster, the quality of preprocessing (cleaning) remains the most critical factor in achieving high precision.

## (8) Student's details
- Name:Reshma Malik Shafaat
- Roll No: 33
- UIN: 231A031
- YEAR: TE-AIDS (Sem 6)
