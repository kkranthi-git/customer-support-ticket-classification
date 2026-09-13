# Customer Support Ticket Classification & Priority Prediction

An end-to-end NLP and Machine Learning project that automatically classifies customer support tickets into the appropriate support queue and predicts ticket priority.

The project uses TF-IDF and Linear SVM models, with a Streamlit application and Docker deployment.

---

## Project Overview

Customer support teams receive large numbers of tickets that need to be routed to the correct department and prioritized efficiently.

This project builds an NLP-based system that performs two tasks:

- Predicts the **support queue** for a customer ticket.
- Predicts the **priority** of the ticket.

---

## Dataset

**Customer IT Support – Ticket Dataset**

- 28,587 tickets
- English and German tickets
- English subset used for modeling
- Subject and ticket body used as text input
- 10 support queue categories
- 3 priority categories
- No duplicate complete rows

> **Note:** The dataset is synthetic and may not fully represent real-world customer support data.

---

## Machine Learning Approach

The project uses the following pipeline:

```text
Customer Ticket
       ↓
Text Preparation
       ↓
TF-IDF Vectorization
       ↓
Linear SVM
      / \
     ↓   ↓
 Queue  Priority
Prediction Prediction
Text Processing
Subject and body are combined.
Missing subjects are handled.
Empty text records are removed.
English tickets are used for modeling.
Models Evaluated
Logistic Regression
Multinomial Naive Bayes
Linear SVM

Linear SVM achieved the best overall performance for both tasks.

Model Performance
Task	Model	Accuracy	Macro F1
Queue Classification	Linear SVM	66.80%	68.61%
Priority Prediction	Linear SVM	69.65%	68.63%

Queue Classification
Accuracy: 66.80%
Macro F1: 68.61%
Best class: Billing and Payments (F1: 0.85)

The main classification challenges occurred between similar categories such as Technical Support, Product Support, and IT Support.

Priority Prediction
Accuracy: 69.65%
Macro F1: 68.63%
Priority	F1 Score
High	0.73
Medium	0.69
Low	0.63
Error Analysis

The main errors in Queue Classification occurred between:

Product Support → Technical Support
Technical Support → IT Support
Technical Support → Product Support
IT Support → Technical Support
Technical Support → Customer Service

These categories contain overlapping technical and support-related terminology.

TF-IDF captures important words and phrases effectively, but it has limitations when deeper semantic understanding is required.

Streamlit Application

A Streamlit application provides a simple interface for making predictions.

Users enter:

Ticket Subject
Ticket Description

The application predicts:

Support Queue
Priority

Example:

Support Queue: Billing and Payments
Priority: HIGH
Project Structure
customer-support-ticket-classification/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── priority_classifier.pkl
│   ├── priority_tfidf_vectorizer.pkl
│   ├── queue_classifier.pkl
│   └── queue_tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 04_queue_modeling.ipynb
│   ├── 05_priority_modeling.ipynb
│   └── 06_error_analysis.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   └── test_prediction.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
Testing

Automated tests are implemented using pytest.

Run tests with:

python -m pytest

Current result:

3 tests passed

The tests verify:

Prediction output structure
Prediction value types
Empty ticket handling

Docker
The application can be run inside a Docker container.

Build
docker build -t customer-support-ticket-classifier .

Run
docker run -p 8501:8501 customer-support-ticket-classifier

Open the application:
http://localhost:8501

Local Setup
Clone the Repository
git clone  https://github.com/kkranthi-git/customer-support-ticket-classification.git
cd customer-support-ticket-classification

Create Virtual Environment

python -m venv .venv

Activate on Windows
.venv\Scripts\activate

Install Dependencies
pip install -r requirements.txt

Run the Application
python -m streamlit run app/streamlit_app.py

Technologies
Python
Pandas
NumPy
Scikit-learn
TF-IDF
Logistic Regression
Multinomial Naive Bayes
Linear SVM
Matplotlib
Seaborn
Streamlit
Docker
Pytest
Joblib
Jupyter Notebook
Git & GitHub

Future Improvements
Transformer-based NLP models
Sentence embeddings
Multilingual classification
Confidence calibration
Hyperparameter optimization
Model monitoring
REST API deployment
CI/CD pipeline
Cloud deployment
Real-world customer support data
Author

Kranthi Kumar

 Data Scientist | Machine Learning | NLP | AI Engineering

Skills: Python • Machine Learning • NLP • SQL • Data Analysis • Scikit-learn • Streamlit • Docker