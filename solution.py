import json
import sys
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
from company_qualifier.retrieval.filters import filter_boolean, filter_country
from company_qualifier.retrieval.fuse import rrf_fuse
from company_qualifier.retrieval.lexical import LexicalSearcher
from company_qualifier.retrieval.search import DenseSearcher

REGIONS = yaml.safe_load(Path("queries/benchmark.yaml").read_text(encoding="utf-8")).get("regions", {})


def qualify(query: str) -> list[dict]:
    companies, _ = load_companies(SRC_DATASET)
    companies, _ = dedupe_exact(companies)
    profiles = dict(zip((c.row_index for c in companies), build_all(companies)))
    countries = {c.row_index: c.address.country_code for c in companies}
    booleans = {c.row_index: {"is_public": c.is_public} for c in companies}
    names = {c.row_index: c.operational_name or f"row {c.row_index}" for c in companies}

    dense, lexical, rr, llm = DenseSearcher(), LexicalSearcher(), Reranker(), OllamaClient()

    plan = llm.complete(PLANNER_PROMPT.format(query=query), QueryPlan)

    codes: list[str] = []
    for c in plan.country_codes:
        codes.extend(REGIONS.get(c, [c]))

    texts = [plan.semantic_text or query] + plan.hyde_descriptions
    rankings = [dense.search(t) for t in texts] + [lexical.search(plan.semantic_text or query)]
    hybrid = rrf_fuse(rankings)
    if codes:
        hybrid = [i for i in hybrid if countries.get(i) in set(codes)]
    hybrid = filter_boolean(hybrid, plan.boolean_filters, booleans)

    reranked = rr.rerank(query, hybrid[:150], top_k=40)
    verdicts = judge_candidates(llm, query, reranked, profiles)

    return [{"row_index": v.row_index, "name": names.get(v.row_index, ""),
             "verdict": v.verdict, "evidence": v.evidence, "reason": v.reason,
             "unsupported_constraints": plan.unsupported}
            for v in verdicts if v.verdict == "match"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    results = qualify(sys.argv[1])
    print(json.dumps(results, indent=2, ensure_ascii=False))