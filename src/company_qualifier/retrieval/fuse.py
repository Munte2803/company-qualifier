from collections import defaultdict


def rrf_fuse(rankings: list[list[int]], k: int = 60, top_k: int = 457) -> list[int]:
    scores: defaultdict[int, float] = defaultdict(float)
    for ranking in rankings:
        for i, row_index in enumerate(ranking):
            scores[row_index] += 1 / (k + i + 1)
    return sorted(scores, key=lambda idx: scores[idx], reverse=True)[:top_k]
    
        