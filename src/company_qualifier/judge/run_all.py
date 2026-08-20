import json
import traceback
from pathlib import Path

import yaml

from company_qualifier.ingest.dedupe import dedupe_exact
from company_qualifier.ingest.parse import load_companies
from company_qualifier.ingest.profile import build_all
from company_qualifier.judge.judge import judge_candidates
from company_qualifier.judge.llm import OllamaClient
from company_qualifier.judge.prompts import PLANNER_PROMPT
from company_qualifier.judge.schemas import QueryPlan
from company_qualifier.rerank.cross_encoder import Reranker
from company_qualifier.retrieval.embed import SRC_DATASET
from company_qualifier.retrieval.filters import filter_country
from company_qualifier.retrieval.filters import filter_boolean
from company_qualifier.retrieval.fuse import rrf_fuse
from company_qualifier.retrieval.lexical import LexicalSearcher
from company_qualifier.retrieval.search import DenseSearcher

BENCHMARK = Path("queries/benchmark.yaml")
OUT_DIR = Path("results/runs/night1")


def main() -> None:
    bench = yaml.safe_load(BENCHMARK.read_text(encoding="utf-8"))
    regions: dict[str, list[str]] = bench.get("regions", {})

    companies, _ = load_companies(SRC_DATASET)
    companies, _ = dedupe_exact(companies)
    profiles = dict(zip((c.row_index for c in companies), build_all(companies)))
    countries = {c.row_index: c.address.country_code for c in companies}
    booleans = {c.row_index: {"is_public": c.is_public} for c in companies}

    dense, lexical, rr, llm = DenseSearcher(), LexicalSearcher(), Reranker(), OllamaClient()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for q in bench["queries"]:
        qid, text = q["id"], q["text"]
        out_path = OUT_DIR / f"{qid}.json"
        if out_path.exists():                    
            print(f"{qid}: exists, skipping")
            continue
        print(f"=== {qid}: {text}")
        try:
            plan = llm.complete(PLANNER_PROMPT.format(query=text), QueryPlan)

            
            codes: list[str] = []
            for c in plan.country_codes:
                codes.extend(regions.get(c, [c]))

                            

            
            texts = [plan.semantic_text or text] + plan.hyde_descriptions
            rankings = [dense.search(t) for t in texts] + [lexical.search(plan.semantic_text or text)]
            hybrid = rrf_fuse(rankings)
            if codes:
                hybrid = filter_country(hybrid, codes[0], countries) if len(codes) == 1 else \
                         [i for i in hybrid if countries.get(i) in set(codes)]
            hybrid = filter_boolean(hybrid, plan.boolean_filters, booleans)

            reranked = rr.rerank(text, hybrid[:150], top_k=40)
            verdicts = judge_candidates(llm, text, reranked, profiles)

            out_path.write_text(json.dumps({
                "query_id": qid, "query": text,
                "plan": plan.model_dump(),
                "candidates_after_rerank": reranked,
                "verdicts": [v.model_dump() for v in verdicts],
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{qid}: saved {len(verdicts)} verdicts")
        except Exception:
            print(f"{qid}: FAILED")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()