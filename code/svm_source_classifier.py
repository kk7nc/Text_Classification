# -*- coding: utf-8 -*-
"""
SVM_source_classifier.py
This code trains a SVM model to classify the source of a text document
based on writing style and topic using the fetch_20newsgroups dataset.
Removes headers, footers, and quotes to prevent data leakage.
Author: Jane Heng
Date: Oct.10.2025
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn import metrics

def load_data(categories):
    """
    Loads training and testing data from the 20 Newsgroups dataset for the specified categories.
    Removes headers, footers, and quotes to reduce noise and prevent data leakage.

    Args:
        categories (list of str): List of category names to include.

    Returns:
        tuple: A tuple containing:
            - train_data (list of str): Training text samples.
            - train_target (list of int): Labels for training samples.
            - test_data (list of str): Testing text samples.
            - test_target (list of int): Labels for testing samples.
            - target_names (list of str): Names of the target categories.
    """
    train_data = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )
    test_data = fetch_20newsgroups(
        subset='test',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )
    return train_data.data, train_data.target, test_data.data, test_data.target, train_data.target_names

def train_model(X_train, y_train):
    """
    Trains a Support Vector Machine (SVM) classifier using TF-IDF features.

    Args:
        X_train (list of str): Training text data.
        y_train (list of int): Corresponding labels for training data.

    Returns:
        sklearn.pipeline.Pipeline: A trained pipeline combining TF-IDF vectorization and SVM classification.
    """
    model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, target_names):
    """
    Evaluates the trained model on the test data and prints a classification report.

    Args:
        model (Pipeline): Trained model pipeline.
        X_test (list of str): Test text data.
        y_test (list of int): True labels for test data.
        target_names (list of str): Names of the target categories.

    Returns:
        None
    """
    predicted = model.predict(X_test)
    print(metrics.classification_report(y_test, predicted, target_names=target_names))

def run_source_classification(categories):
    """
    Executes the full classification pipeline:
    - Loads data for the specified categories
    - Trains an SVM model
    - Evaluates model performance

    Args:
        categories (list of str): List of category names to classify.

    Returns:
        sklearn.pipeline.Pipeline: The trained model pipeline.
    """
    X_train, y_train, X_test, y_test, target_names = load_data(categories)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, target_names)
    return model

if __name__ == "__main__":
    categories = ['talk.politics.misc', 'rec.sport.hockey', 'sci.space']
    run_source_classification(categories)