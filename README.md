#  Sentiment Analysis of Emerging Tech Trends: The SolarGlass Smartphone

## (1) Problem Statement
Social media platforms like Twitter (X) generate vast amounts of unstructured data. For a tech company launching a new product like the SolarGlass Smartphone, it is nearly impossible to manually track every user's opinion. This project addresses the need for an automated system to categorize public sentiment to understand market reception.

## (2) Objective
- To collect and manually label a dataset of 100 tweets regarding "SolarGlass" technology.
- To implement and compare machine learning classifiers (Naive Bayes, Logistic Regression and SVM).
- To evaluate models using Precision, Recall, and Accuracy to determine the most effective approach for sentiment detection.

## (3) Dataset
- Source: Manually curated dataset of 100 tweets related to the "SolarGlass Smartphone" (2026 Buzzword).
- Features: Tweet_Text (Raw text), Label (Categorical: 1 for Positive, 0 for Neutral, -1 for Negative), cleaned_text (lowercased text with punctuation removed).
- Size: 100 rows (80 for training, 20 for testing).

## (4) Methodology
1. Data Preprocessing: Converted text to lowercase, used Regular Expressions (re library) to remove punctuation, URLs, and stopwords to reduce noise.
2. EDA: Analyzed the distribution of sentiments (Positive vs. Negative vs. Neutral).
3. Model Building: Utilized CountVectorizer for feature extraction and implemented three classifiers: Multinomial Naive Bayes, Logistic Regression, and Support Vector Machine (SVM).
4. Evaluation: Used a test set of 20 tweets to generate a classification_report and a Confusion Matrix.

## (5) Results
Insights:
- Best Model: Logistic Regression performed best with 80% accuracy.
- It achieved a perfect Precision (1.00) for Positive sentiments, meaning it never falsely labeled a neutral/negative tweet as positive.
- All models showed high recall for Negative sentiments, effectively identifying customer complaints.
- **View the full technical report:** [Download Sentiment Analysis Report PDF](./report/Sentiment_Analysis_Report.pdf)

## (6) How to Run
```bash
pip install -r requirements.txt
python main.py
```

## (7) Conclusion
Logistic Regression proved to be the most robust classifier for this small-scale text dataset. The project highlights that while simple models work well, the quality of text cleaning significantly impacts the precision of sentiment detection in niche tech topics.
The project successfully demonstrates that even with a small dataset of 100 tweets, machine learning models can effectively categorize sentiment. While Naive Bayes is faster, the quality of preprocessing (cleaning) remains the most critical factor in achieving high precision.

## (8) Student's details
- Name:Reshma Malik Shafaat
- Roll No: 33
- UIN: 231A031
- YEAR: TE-AIDS (Sem 6)
