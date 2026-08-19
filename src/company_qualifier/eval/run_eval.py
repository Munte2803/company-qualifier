from company_qualifier.eval.metrics import (
    load_golden, ndcg_at_k, precision_at_k, recall_at_k,
)
from company_qualifier.ingest.dedupe import dedupe_exact
from company_qualifier.rerank.cross_encoder import Reranker
from company_qualifier.retrieval.embed import SRC_DATASET
from company_qualifier.ingest.parse import load_companies
from company_qualifier.retrieval.filters import filter_country
from company_qualifier.retrieval.fuse import rrf_fuse
from company_qualifier.retrieval.search import DenseSearcher
from company_qualifier.retrieval.lexical import LexicalSearcher

QUERIES = {
    "q01": ( "Logistic companies in Romania", "ro"),
    "q06": ( "Pharmaceutical companies in Switzerland", "ch"),
    "q12": ( "Critical components for EV battery production", None),
}
K = 10
K2=150
K3=40


def main() -> None:
    golden = load_golden("data/golden/golden.tsv")
    dense = DenseSearcher()
    lexical = LexicalSearcher()
    rr = Reranker()
    companies, _ = load_companies(SRC_DATASET)
    companies, _ = dedupe_exact(companies)
    countries = {c.row_index: c.address.country_code for c in companies}

    print(f"{'query':6} {'config':10} {'P@10':>6} {'R@10':>6} {'nDCG@10':>8} {'R@150':>6} {'R@40':>6}")

    for qid, (text, country_code) in QUERIES.items():
        dense_ranking = dense.search(text)
        lexical_ranking = lexical.search(text)
        hybrid   = rrf_fuse([dense_ranking, lexical_ranking])        
        if country_code is not None:
            hybrid = filter_country(hybrid, country_code, countries)
        g = golden[qid]
        reranked = rr.rerank(text, hybrid[:150], top_k=40)
        for config, ranking in [("baseline", dense_ranking), ("hybrid+f", hybrid), ("hybrid+f+rr", reranked)]:
            r150 = (f"{recall_at_k(ranking, g, K2):>6.2f}"
                    if len(ranking) >= K2 else f"{'—':>6}")
            print(f"{qid:6} {config:10} "
                  f"{precision_at_k(ranking, g, K):>6.2f} "
                  f"{recall_at_k(ranking, g, K):>6.2f} "
                  f"{ndcg_at_k(ranking, g, K):>8.2f} "
                  f"{r150} "
                  f"{recall_at_k(ranking, g, K3):>6.2f}")

if __name__ == "__main__":
    main()