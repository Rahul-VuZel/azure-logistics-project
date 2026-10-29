# preprocessing.py
# This file CLEANS and PREPARES our raw data for machine learning.
# Raw data straight from the source is messy - this file makes it usable.
#
# Concepts used from the course:
#   Week 5: TF-IDF vectorization for text (sentiment analysis on customer feedback)
#   Week 6: StandardScaler + 70/30 train/test split (breast cancer evaluation project)
#
# The order always matters:
#   1. Clean the text (remove noise)
#   2. Encode labels (turn words into numbers)
#   3. Split into train/test (so we don't cheat by peeking at test data)
#   4. Scale the features (so no single feature unfairly dominates)
#   5. Save the results (so we don't have to redo this every run)

import re
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fix for Windows: make sure Python can find config.py in the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import TEST_SIZE, RANDOM_STATE

# Where to save our cleaned, ready-to-use data
PROCESSED_DIR = "data/02_processed"


# TEXT CLEANING  (Week 5 - NLP Sentiment Analysis)

def clean_text(text):
    """
    Clean a single piece of text so it is ready for TF-IDF.
    """
    # Make everything lowercase
    text = text.lower()

    # Keep only letters, numbers, and spaces - remove everything else
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Collapse multiple spaces into one and trim the edges
    text = re.sub(r"\s+", " ", text).strip()

    return text


def vectorize_tfidf(texts, max_features=5000):
    """
    Convert a list of text strings into a numeric matrix using TF-IDF.
    Arguments:
        texts        : a list of strings 
        max_features : how many unique words to track (default 5000)

    Returns:
        X          : numeric matrix (rows = documents, columns = word scores)
        vectorizer : the fitted TF-IDF object 
    """
    print("\n[Preprocessing] Running TF-IDF vectorization on " + str(len(texts)) + " texts")

    # Create and fit the vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)

    print("[Preprocessing] TF-IDF done. Output matrix shape: " + str(X.shape))
    return X, vectorizer


# LABEL ENCODING

def encode_labels(series):
    """
    Convert text labels into numbers the model can understand.

    For our logistics dataset:
        "Delayed"  →  1   (the problem case we want to catch)
        "On-Time"  →  0   (normal/good case)

    This is the same concept from Week 6 where:
        "M" (Malignant)  →  1
        "B" (Benign)     →  0

    Arguments:
        series : a pandas Series containing the label column

    Returns:
        y  : array of 0s and 1s
        le : the LabelEncoder (save this to convert predictions back to text later)
    """
    le = LabelEncoder()
    y = le.fit_transform(series)

    mapping = {}
    for label in le.classes_:
        mapping[label] = int(le.transform([label])[0])
    print("[Preprocessing] Label mapping: " + str(mapping))

    return y, le


# TRAIN / TEST SPLIT

def split_data(X, y, test_size=TEST_SIZE):
    """
    Split data into a training set and a testing set.

    Arguments:
        X         : feature matrix
        y         : labels array
        test_size : fraction for testing (default 0.30 from config.py)

    Returns:
        X_train, X_test, y_train, y_test
    """
    train_pct = int((1 - test_size) * 100)
    test_pct = int(test_size * 100)
    print("\n[Preprocessing] Splitting data: " + str(train_pct) + "% train / " + str(test_pct) + "% test")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,   # fixed seed = reproducible results every run
        stratify=y                   # keep label balance in both splits
    )

    print("[Preprocessing] Training samples: " + str(len(X_train)))
    print("[Preprocessing] Testing samples:  " + str(len(X_test)))
    return X_train, X_test, y_train, y_test


# FEATURE SCALING

def scale_features(X_train, X_test):
    """

    Arguments:
        X_train : training feature matrix
        X_test  : testing feature matrix

    Returns:
        X_train_scaled : scaled training features
        X_test_scaled  : scaled testing features (using training mean/std)
        scaler         : the fitted StandardScaler object
    """
    print("[Preprocessing] Scaling features (fitting scaler on training data only)...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # learn mean and std from train only
    X_test_scaled = scaler.transform(X_test)          # apply the same scale to test

    print("[Preprocessing] Scaling complete.")
    return X_train_scaled, X_test_scaled, scaler

# FULL PREPROCESSING PIPELINE  (renamed from preprocess_breast_cancer)

def preprocess_logistics_data(df):
    """
    Run the FULL preprocessing pipeline on our logistics dataset.

    This function chains together all the steps above in the correct order.
    Call this one function and it handles everything automatically.

    Steps:
        1. Drop ID column (it is just a row number, useless for ML)
        2. Encode labels  (Delayed=1, On-Time=0)
        3. Split data     (70% train, 30% test)
        4. Scale features (zero mean, unit variance)
        5. Save processed arrays to data/02_processed/
    """
    print("RUNNING PREPROCESSING PIPELINE")

    # Drop the ID column - it is just a row identifier, not a useful feature
    df = df.drop(columns=["id"], errors="ignore")
    print("[Preprocessing] Dropped 'id' column (not a useful feature)")

    # Encode labels (turn text labels into 0s and 1s)
    y, le = encode_labels(df["diagnosis"])
    X = df.drop(columns=["diagnosis"]).values.astype(float)
    print("[Preprocessing] Feature matrix shape: " + str(X.shape))

    # Split into training and testing sets (70/30)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale the features (prevent data leakage - fit on train only!)
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Save the processed arrays so we don't have to rerun this every time
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)
    print("[Preprocessing] Saved processed arrays to: " + PROCESSED_DIR + "/")

    # Package everything into one dictionary for easy passing to model.py
    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "scaler":        scaler,
        "label_encoder": le,
    }
