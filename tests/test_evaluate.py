import numpy as np

from fraud_engine.evaluate import (best_threshold, chronological_split, cost_at_threshold,
                                   precision_recall_at)


def test_split_is_chronological_and_disjoint(feat):
    train, val, test = chronological_split(feat)
    assert train["ts"].max() < val["ts"].min() <= val["ts"].max() < test["ts"].min()
    assert len(train) + len(val) + len(test) == len(feat)


def test_precision_recall_at_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([.9, .8, .1, .2, .3, .1, .2, .3, .1, .2])
    assert precision_recall_at(y, score, 0.2) == (1.0, 1.0, 2)


def test_best_threshold_flags_only_when_worth_it():
    y = np.array([1, 0, 0, 0])
    score = np.array([.9, .8, .1, .1])
    cheap = best_threshold(y, score, np.array([1000.0, 50, 50, 50]), review_cost=10)
    dear = best_threshold(y, score, np.array([100.0, 50, 50, 50]), review_cost=500)
    assert cheap["alerts"] == 1 and cheap["saved"] > 0      # a 10 review to stop a 1000 fraud is worth it
    assert dear["alerts"] == 0 and dear["saved"] == 0       # a 500 review of a 100 fraud is not


def test_cost_at_threshold_arithmetic():
    y = np.array([1, 1, 0])
    amount = np.array([100.0, 200.0, 50.0])
    r = cost_at_threshold(y, np.array([.9, .1, .8]), amount, 0.5, review_cost=10)
    assert r["alerts"] == 2 and r["fraud_caught"] == 1
    assert r["amount_missed"] == 200.0 and r["cost"] == 220.0
    assert r["baseline_cost"] == 300.0 and r["saved"] == 80.0
