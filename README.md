# 🕵️ Fake News Detection System

A Streamlit web application that detects fake news using both Logistic Regression and Deep Learning (BiLSTM + Conv1D) models trained on multiple datasets.

![Fake News Detection Demo](https://media.giphy.com/media/v1.Y2lkPWVjZjA1ZTQ3cnplcHM3aDBwZW00bDR2ZGZjM3NsOHBod3U2M2s0OXZybzM5NHhuaCZlcD12MV9naWZzX3JlbGF0ZWQmY3Q9Zw/fSplBjxrvJAnP1W26q/giphy.gif) 

## 📌 Features

- **Dual-model prediction system**:
  - Logistic Regression with TF-IDF vectorization
  - TensorFlow Sequential model with Conv1D + BiLSTM
- **Probability-based outputs** showing likelihood of news being fake or real
- **Trained on multiple datasets** for better generalization
- **Interactive UI** with a clean layout using Streamlit

---

## 📂 Repository Structure

```
.
├── app.py                  # Streamlit application
├── models/                 # Saved models and vectorizers
│   ├── logistic_model.joblib
│   ├── sequential_model.keras
│   ├── tfidf_vectorizer.joblib
│   └── tokenizer.pickle
├── data/                   # Training datasets
│   ├── True.csv
│   └── Fake.csv
│   └── news_dataset.csv
├── notebook/              # Jupyter notebooks for analysis
│   ├── Study.ipynb
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

To run the Streamlit app locally:

```bash
streamlit run app.py
```

The app will open in your browser at [http://localhost:8501](http://localhost:8501).  
You can enter a news headline or content and choose a model for prediction.

---

## 🧠 Models Overview

### 1. Logistic Regression with TF-IDF
- Scikit-learn implementation
- Fast and lightweight
- TF-IDF for feature extraction

### 2. BiLSTM + Conv1D Deep Learning Model
- Built with TensorFlow/Keras
- **Architecture**:
  - Embedding Layer
  - Conv1D Layer for local pattern detection
  - Bidirectional LSTM for sequence modeling
  - Dense layers with dropout for regularization

---

## 📊 Datasets

- **Fake News Detection Dataset** (Kaggle)
  - Balanced dataset labeled as REAL or FAKE
- **Indian Fake News Dataset** (Kaggle)
  - Focuses on Indian context, includes regional samples

---

## 📈 Model Performance

| Model                 | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 92.3%    | 91.8%     | 93.1%  | 92.4%    |
| BiLSTM + Conv1D      | 94.7%    | 94.2%     | 95.3%  | 94.7%    |

---

## 🧪 Suggested Additions

- Add GIF/screen recording of app in action

- Show confusion matrices and sample predictions in the notebooks

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit an issue or pull request.

---
<!--
## 📜 License

Licensed under the MIT License. See `LICENSE` for details. -->
