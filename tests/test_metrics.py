import numpy as np
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from evaluate import macro_f1, kappa, fleiss_kappa, full_metrics, bootstrap_ci

Y = [1, 2, 3, 4, 4, 3, 2, 1, 2, 3]
P = [1, 2, 3, 3, 4, 0, 3, 1, 2, 4]


def test_against_sklearn():
    assert np.isclose(macro_f1(Y, P), f1_score(Y, P, labels=[1, 2, 3, 4], average="macro"))
    m = full_metrics(Y, P)
    assert np.isclose(m["accuracy"], accuracy_score(Y, P))
    assert np.isclose(kappa(Y, P), cohen_kappa_score(Y, P, labels=[0, 1, 2, 3, 4]))
    assert np.isclose(kappa(Y, P, "linear"), cohen_kappa_score(Y, P, labels=[0, 1, 2, 3, 4], weights="linear"))
    assert m["invalid_count"] == 1
    assert np.isclose(m["under_rate"], 0.1) and np.isclose(m["over_rate"], 0.2)  # 4->3 недооценка; 2->3, 3->4 переоценка; 3->0 не считается
    assert len(m["confusion"]["matrix"]) == 4 and len(m["confusion"]["matrix"][0]) == 5


def test_perfect():
    m = full_metrics(Y, Y)
    assert m["accuracy"] == 1 and m["macro_f1"] == 1 and m["kappa_linear"] == 1


def test_fleiss():
    assert fleiss_kappa([[3, 0], [0, 3], [3, 0]]) == 1.0
    M = [[0,0,0,0,14],[0,2,6,4,2],[0,0,3,5,6],[0,3,9,2,0],[2,2,8,1,1],[7,7,0,0,0],[3,2,6,3,0],[2,5,3,2,2],[6,5,2,1,0],[0,2,2,3,7]]
    assert abs(fleiss_kappa(M) - 0.210) < 0.001


def test_bootstrap_reproducible():
    a = bootstrap_ci(Y, P, 200, 42); b = bootstrap_ci(Y, P, 200, 42)
    assert a == b and a["accuracy"][0] <= a["accuracy"][1]
