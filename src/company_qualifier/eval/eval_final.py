import json
from pathlib import Path

from company_qualifier.eval.metrics import (
    load_golden, ndcg_at_k, precision_at_k, recall_at_k,
)

RUNS = Path("results/runs/night1")
K = 10


def main() -> None:
    golden = load_golden("data/golden/golden.tsv")

    print(f"{'query':6} {'P@10':>6} {'R@10':>6} {'nDCG@10':>8} {'#match':>7}")
    for qid in ["q01", "q06", "q12"]:
        data = json.loads((RUNS / f"{qid}.json").read_text(encoding="utf-8"))
        matches = [v["row_index"] for v in data["verdicts"] if v["verdict"] == "match"]
        g = golden[qid]
        print(f"{qid:6} {precision_at_k(matches, g, K):>6.2f} "
              f"{recall_at_k(matches, g, K):>6.2f} "
              f"{ndcg_at_k(matches, g, K):>8.2f} {len(matches):>7}")


if __name__ == "__main__":
    main()