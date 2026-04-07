# Quora Duplicate Question Detection

This project focuses on identifying whether two questions are semantically similar or duplicates using classical machine learning techniques.

## 🚀 Overview
Duplicate questions are a common issue in platforms like Quora. This project builds an NLP pipeline to detect semantic similarity between question pairs using feature engineering and traditional ML models.

## 🧠 Approach

### 1. Text Preprocessing
- Lowercasing, punctuation removal
- Tokenization
- Stopword handling

### 2. Feature Engineering
- TF-IDF (unigrams + bigrams)
- Bag of Words comparison
- N-gram based similarity
- Handcrafted features:
  - Word overlap ratio
  - Length difference
  - Common word count
  - Token-based similarity metrics

### 3. Model Building
Trained and compared multiple models:
- Logistic Regression
- Linear SVM
- XGBoost

### 4. Hyperparameter Tuning
- Used Optuna for automated tuning
- Optimized parameters for best performance

### 5. Evaluation
- Accuracy Score used for evaluation
- Final accuracy achieved: **0.869**

## 📊 Key Learnings
- Feature engineering significantly improves performance
- Model comparison is crucial — no single model always wins
- Avoiding data leakage is critical for reliable results

## 🛠️ Tech Stack
- Python
- Scikit-learn
- Pandas, NumPy
- Optuna
- XGBoost

## 📌 Future Improvements
- Add semantic embeddings (Word2Vec / BERT)
- Deploy as a web application
- Improve generalization on unseen data
