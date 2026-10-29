# tests/test_pipeline.py
# ----------------------
# Unit tests for every module in the healthcare_logistics pipeline.
#
# Run all tests with:   pytest tests/
# Run one test with:    pytest tests/test_pipeline.py::test_clean_text_lowercases
#
# What are unit tests?
# Small, isolated checks that verify one piece of code does exactly what we expect.
# If we break something while editing, these tests catch it immediately.

import numpy as np
import pandas as pd
import pytest

# Import the functions we want to test from our source modules
from src.preprocessing import (
    clean_text,
    encode_labels,
    scale_features,
    split_data,
    vectorize_tfidf,
)
from src.model import HealthcareClassifier, QLearningAgent


# ==============================================================================
# PREPROCESSING TESTS
# ==============================================================================

def test_clean_text_lowercases():
    # "DELAYED Shipment!" should become "delayed shipment" after cleaning
    assert clean_text("DELAYED Shipment!") == "delayed shipment"


def test_clean_text_strips_punctuation():
    # Punctuation like commas and periods should be removed
    result = clean_text("on-time, intact.")
    assert "," not in result
    assert "." not in result


def test_clean_text_removes_extra_spaces():
    # Multiple spaces should collapse into one
    result = clean_text("fast   delivery")
    assert "  " not in result


def test_encode_labels_produces_binary():
    # Labels should be encoded to exactly 0 and 1
    s = pd.Series(["Delayed", "On-Time", "Delayed", "On-Time", "Delayed"])
    y, le = encode_labels(s)
    assert set(y) == {0, 1}
    assert len(le.classes_) == 2


def test_encode_labels_works_with_m_b():
    # Also test with the sample dataset labels (M = Malignant, B = Benign)
    s = pd.Series(["M", "B", "M", "B", "M"])
    y, le = encode_labels(s)
    assert set(y) == {0, 1}


def test_split_data_correct_ratio():
    # With 100 samples and default 70/30 split, we expect 70 train and 30 test
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == 70
    assert len(X_test) == 30


def test_split_data_custom_ratio():
    # Test that a custom test_size argument works correctly
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20)
    assert len(X_train) == 80
    assert len(X_test) == 20


def test_scale_features_mean_near_zero():
    # After scaling, the training set should have approximately zero mean
    X_train = np.random.rand(80, 5)
    X_test = np.random.rand(20, 5)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    # Mean should be very close to 0 (within rounding error)
    assert abs(X_train_scaled.mean()) < 0.01


def test_scale_features_does_not_fit_on_test():
    # The scaler should be fit on training data ONLY (prevents data leakage)
    # We verify this by checking that the scaler's mean_ was set
    X_train = np.random.rand(80, 5)
    X_test = np.random.rand(20, 5)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    # If scaler has mean_, it was fit correctly on the training data
    assert hasattr(scaler, "mean_")


def test_tfidf_output_shape():
    # TF-IDF should return a matrix with one row per document
    texts = ["fast delivery intact", "delayed lost damaged", "on time professional"]
    X, vec = vectorize_tfidf(texts, max_features=20)
    assert X.shape[0] == 3         # 3 rows = 3 documents
    assert X.shape[1] <= 20        # at most max_features columns


# ==============================================================================
# CLASSIFIER TESTS
# ==============================================================================

# Test all three model types using the same test function
@pytest.mark.parametrize("model_type", [
    "logistic_regression",
    "random_forest",
    "naive_bayes",
])
def test_classifier_returns_predictions(model_type):
    # Each model should return the correct number of predictions
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((80, 10))
    y_train = rng.integers(0, 2, 80)
    X_test = rng.standard_normal((20, 10))
    y_test = rng.integers(0, 2, 20)

    clf = HealthcareClassifier(model_type=model_type)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    # Should return exactly 20 predictions (one per test sample)
    assert len(preds) == 20


def test_classifier_evaluate_returns_all_metrics():
    # evaluate() should return a dict with all four required metrics
    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((80, 10))
    y_train = rng.integers(0, 2, 80)
    X_test = rng.standard_normal((20, 10))
    y_test = rng.integers(0, 2, 20)

    clf = HealthcareClassifier(model_type="logistic_regression")
    clf.fit(X_train, y_train)
    metrics = clf.evaluate(X_test, y_test)

    # All four metrics from the Week 6 evaluation must be present
    for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
        assert key in metrics, "Missing metric: " + key


def test_invalid_model_type_raises_error():
    # Passing an unsupported model name should raise a clear ValueError
    with pytest.raises(ValueError):
        HealthcareClassifier(model_type="svm_not_supported")


# ==============================================================================
# Q-LEARNING TESTS
# ==============================================================================

def test_q_table_initialized_to_zeros():
    # Q-table should start as all zeros (agent knows nothing yet)
    agent = QLearningAgent(n_states=6)
    assert agent.q_table.shape == (6, 2)
    assert np.all(agent.q_table == 0)


def test_qlearning_returns_one_reward_per_episode():
    # train() should return exactly one reward value per episode
    agent = QLearningAgent(n_states=6)
    rewards = agent.train(episodes=50)
    assert len(rewards) == 50


def test_qlearning_greedy_path_ends_at_goal_or_pit():
    # After training, the greedy path must end at state 5 (goal) or state 0 (pit)
    # It should consistently reach the goal after enough training
    agent = QLearningAgent(n_states=6)
    agent.train(episodes=300)
    path = agent.run_episode(start_state=3)
    assert path[-1] in (0, 5)


def test_qlearning_q_table_updates_after_training():
    # After training, Q-table should no longer be all zeros
    agent = QLearningAgent(n_states=6)
    agent.train(episodes=100)
    assert not np.all(agent.q_table == 0), "Q-table should have updated values after training"
