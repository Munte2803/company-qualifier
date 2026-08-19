import math

from ranx import Qrels, Run, evaluate

from company_qualifier.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k

GOLDEN_Q = {1: "relevant", 2: "partial", 3: "irrelevant", 4: "relevant"}
RANKING = [1, 3, 2, 5]

def test_precision():
    assert precision_at_k(RANKING, GOLDEN_Q, 4) == 0.5

def test_recall():
    assert math.isclose(recall_at_k(RANKING, GOLDEN_Q, 4), 2/3)

def test_ndcg():
    idcg = 2/math.log2(2) + 2/math.log2(3) + 1/math.log2(4)
    assert math.isclose(ndcg_at_k(RANKING, GOLDEN_Q, 4), 2.5/idcg)


def test_ndcg_matches_ranx():
    """Verificare externă: implementarea proprie contra ranx, pe cazul standard."""
    qrels = Qrels({"q_test": {"1": 2, "2": 1, "3": 0, "4": 2}})
    run = Run({"q_test": {str(idx): float(len(RANKING) - i)
                          for i, idx in enumerate(RANKING)}})
    ranx_score = float(evaluate(qrels, run, "ndcg@4"))
    assert math.isclose(ndcg_at_k(RANKING, GOLDEN_Q, 4), ranx_score, rel_tol=1e-6)