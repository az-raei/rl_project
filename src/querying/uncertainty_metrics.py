"""Uncertainty of an ensemble of preference predictors, from their probabilities P(i preferred).

P has shape (M, n): M ensemble members x n candidate pairs.
  ensemble_variance   variance across members of the preference probability   (acquisition score)
  predictive_entropy  binary entropy (nats) of the ensemble-mean probability  (acquisition score / tracked metric)
  mutual_information  entropy of the mean minus mean member entropy (BALD): disagreement, not noise
"""
import numpy as np


def binary_entropy(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def ensemble_variance(P) -> np.ndarray:
    return np.var(np.asarray(P, dtype=np.float64), axis=0)


def predictive_entropy(P) -> np.ndarray:
    return binary_entropy(np.mean(np.asarray(P, dtype=np.float64), axis=0))


def mutual_information(P) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    return binary_entropy(P.mean(axis=0)) - binary_entropy(P).mean(axis=0)
