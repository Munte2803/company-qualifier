import csv
import math
from collections import defaultdict


def load_golden(path: str) -> dict[str, dict[int, str]]:

    golden: defaultdict[str, dict[int, str]] = defaultdict(dict)

    with open(path, encoding='utf-8') as f:          
        reader = csv.DictReader(f, delimiter='\t') 
        for row in reader:
            qid = row["query_id"]                        
            idx = int(row["row_index"])           
            label = row["label"]                      
            golden[qid][idx] = label

    return dict(golden)

def precision_at_k(predictions: list[int], golden_q: dict[int, str], k: int) -> float:
    
    correct = 0
    for idx in predictions[:k]:
        if idx in golden_q and golden_q[idx] in ("relevant", "partial"):
            correct += 1
    return correct / k

def recall_at_k(predictions: list[int], golden_q: dict[int, str], k: int) -> float:
    
    relevant_count = sum(1 for label in golden_q.values() if label in ("relevant", "partial"))
    if relevant_count == 0:
        return 1.0

    correct = 0
    for idx in predictions[:k]:
        if idx in golden_q and golden_q[idx] in ("relevant", "partial"):
            correct += 1
    return correct / relevant_count

def ndcg_at_k(predictions: list[int], golden_q: dict[int, str], k: int) -> float:

    GAINS = {"relevant": 2, "partial": 1}

    gains = [GAINS.get(golden_q.get(idx,""), 0) for idx in predictions[:k]]
    ideal = sorted((GAINS.get(l, 0) for l in golden_q.values()), reverse=True)[:k]
        
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))   

    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 1.0