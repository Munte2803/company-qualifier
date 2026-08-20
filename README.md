# Company Qualification Pipeline

Qualifies companies from a dataset against natural-language queries through a
five-stage funnel: LLM query planning, hybrid retrieval (dense + BM25 + RRF),
hard filters, cross-encoder reranking, and per-candidate LLM judgment with
mandatory evidence. See WRITEUP.md for approach, results and failure modes.

## Setup

1. Python 3.12+, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com).
2. `uv sync`
3. `ollama pull llama3.1:8b`
4. Place the provided dataset at `data/raw/companies.jsonl` (not distributed
   with this repository).
5. Precompute embeddings once: `uv run python -m company_qualifier.retrieval.embed`
   (first run downloads bge-base-en-v1.5 and bge-reranker-base from HuggingFace,
   ~1.5 GB total).

## Run

Single query:

    uv run python solution.py "Logistic companies in Romania"

Full benchmark (12 queries, results in results/runs/night1/):

    uv run python -m company_qualifier.judge.run_all

Evaluation against the golden set:

    uv run python -m company_qualifier.eval.run_eval      # retrieval configs
    uv run python -m company_qualifier.eval.eval_final    # full pipeline
    uv run pytest tests/                                   # metric correctness

## Notes

- Everything runs on CPU. LLM generation dominates latency (~20 min/query on
  a modest CPU); the LLM client is provider-agnostic, so a hosted API drops
  end-to-end time to under a minute per query with no code changes.
- Precomputed artifacts (embeddings, golden labels, benchmark verdicts) are
  committed, so evaluation reproduces without rerunning the LLM stages.