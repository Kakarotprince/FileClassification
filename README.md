# 🧠 Document Classification using Machine Learning

This project is an intelligent document classification system that uses a range of machine learning models, traditional text vectorization techniques (TF-IDF), and modern embeddings (Word2Vec & Doc2Vec) to accurately classify text documents into predefined categories. Ensemble techniques like **hard voting** and **soft voting** are used to improve performance by combining multiple models.

---

## 📁 Datasets Used

We combined two existing datasets to build a richer and more diverse text classification corpus:

1. **[News Article Category Dataset](https://www.kaggle.com/datasets/timilsinabimal/newsarticlecategories)**  
2. **[Text Document Classification Dataset](https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset)**  

These datasets were mapped into unified categories such as:
- News & Current Affairs
- Business & Finance
- Science & Technology
- Arts & Entertainment
- Education & Academia
- Sports

---

## 🧩 Word Embedding

We used the **pre-trained Google News Word2Vec model** (`GoogleNews-vectors-negative300.bin`) for document vectorization:

📥 Download it here: [Google News Word2Vec Embeddings](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300)

---

## 🚀 Project Pipeline

1. **Data Loading & Preprocessing**
2. **Category Mapping & Merging Datasets**
3. **Text Cleaning, Tokenization, and Lemmatization**
4. **Vectorization:**
   - TF-IDF
   - Word2Vec
   - Doc2Vec
5. **Oversampling for Class Imbalance (SMOTE, ADASYN, Random Oversampling)**
6. **Model Training:**
   - Naive Bayes
   - SVM
   - Random Forest
   - AdaBoost
   - XGBoost
   - Word2Vec + Logistic Regression
   - Doc2Vec + Logistic Regression
7. **Model Evaluation**
8. **Ensemble Voting (Hard & Soft)**
9. **Deployment via Streamlit Interface**

---

## 📦 First-Time Setup Instructions

```bash
# Clone the repository
git clone https://github.com/Kakarotprince/FileClassification.git
cd FileClassification

# Install dependencies
pip install -r Requirements.txt
```

### ⚠️ Important Note:

- On **first run**, the Google News vectors need to be loaded and **vector cache saved**.
- **Subsequent runs** will **reuse the saved vectors** to save time and memory.

---

## 📊 Models and Techniques

- **Vectorizers:** TF-IDF, Word2Vec (Google News), Doc2Vec
- **Classifiers:** SVM, RandomForest, AdaBoost, XGBoost, Naive Bayes, Logistic Regression
- **Imbalanced Data Handling:** SMOTE, ADASYN, Random Oversampling
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score
- **Ensembling:** Hard Voting, Soft Voting (based on individual model accuracies)

---

## 🖼️ Streamlit Web Interface

The project includes a user-friendly Streamlit-based UI where users can upload text or files and receive classification results in real-time.

To launch the app:

```bash
streamlit run app.py
```

---

## 📚 References

- [scikit-learn](https://scikit-learn.org/)
- [Gensim](https://radimrehurek.com/gensim/)
- [Streamlit](https://streamlit.io/)
- [Kaggle: GoogleNews Word2Vec](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300)
- [Kaggle: News Article Categories](https://www.kaggle.com/datasets/timilsinabimal/newsarticlecategories)
- [Kaggle: Text Document Dataset](https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset)

---
