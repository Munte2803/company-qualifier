# Company Qualification Pipeline

## 3.1 Approach

I started by inspecting the data. Most example queries carry structured
constraints (country, public status, size, revenue) that can cut a big part
of the candidates before any LLM sees them. Then I built a test set from
three example queries: 167 company-query pairs labeled by hand as relevant,
partial or irrelevant, with candidates pooled from three independent sources
plus a random control sample to check for blind spots.

A plain embedding baseline works at a basic level but breaks on trickier
queries. On "Logistic companies in Romania" it gets P@10 = 0.30 and puts a
truck manufacturer first: cosine similarity sees the vocabulary of
logistics, not the role. Selling logistics is not the same as having it.

So the system is a five-stage funnel. An LLM translates the query into a
structured Pydantic plan: hard filters that cut obvious junk (country,
public status) and semantic text for similarity search. Retrieval scores all
457 companies twice, dense cosine and BM25 lexical, applies the hard
filters, and fuses the two rankings by rank (RRF). Top 150 go to a
cross-encoder that reads each candidate together with the query, so it
catches meaning, not just shared words. Top 40 survive and get judged one by
one by an LLM, which is affordable exactly because the funnel cut 457 down
to 40. Each verdict comes with a quote from the profile as evidence.

The ordering follows the cost of errors: a wrong exclusion is permanent, a
wrong inclusion gets rejected cheaply later. So the top of the funnel
optimizes recall (R@150 = 1.00 on every measured query) and the bottom
optimizes precision. The planner cost doesn't depend on corpus size and
judgment is bounded by the funnel width, so this holds at 100k companies
too, with the hard filters moving in front of the index.

## 3.2 Data

The raw file has 477 JSONL rows: name, address, description, business model,
employee count, revenue, public status, founding year, NAICS codes, target
markets, core offerings.

The rows are not uniform. Some records have compound fields (address, NAICS)
saved as Python repr strings with single quotes instead of JSON, and a few
use an extended schema with extra address fields. I parse tolerantly: fix
what I can, count and report what fails, never crash on one bad row.
Everything goes through Pydantic, so a broken record is a counted event, not
silent corruption.

20 rows are byte-identical duplicates. I remove them by fingerprinting the
content (without the row index), which leaves 457 companies. Companies with
the same name but different content (an oil company twice, an HR platform
four times) stay in: they might carry extra information, and merging them is
a display decision, not an ingestion one.

Every company gets one canonical text profile (name, NAICS label,
description, offerings, markets). All text stages use the same profile:
embeddings, BM25, cross-encoder, LLM judgment. I measured profile length
against the strictest consumer, the cross-encoder with its 512-token limit:
the longest profile is 410 tokens, so nothing gets truncated. Measured, not
assumed.

For evaluation, the two country queries use the complete universe as pool
(all 26 Romanian and all 43 Swiss companies), so recall there is absolute.
The open battery query pools candidates from keyword search, NAICS codes,
and a random sample of 25 as a blind-spot check. Ambiguous groups got
explicit rules applied the same way everywhere: contract manufacturers count
as pharma because they make medicine no matter whose brand it is;
battery-cell makers count as partial because they buy components, they don't
supply them.

## 3.3 Implementation

Everything runs locally on CPU. Models: bge-base-en-v1.5 for embeddings,
bge-reranker-base as cross-encoder, llama3.1:8b through Ollama for planning
and judgment, temperature 0 everywhere.

Embeddings are computed once offline for all 457 profiles and saved to disk
with a row-index mapping, because dedup leaves gaps in the numbering.
Vectors are L2-normalized, so cosine similarity is just one matrix-vector
product at query time. BM25 runs over the same profiles with stemming; the
prefix tokens shared by all profiles have near-zero IDF, so they add no
noise.

The two rankings are fused by rank, not score: BM25 scores are unbounded and
depend on the corpus, cosine lives in [-1, 1], adding them makes no sense.
RRF needs no tuning, takes any number of ranking sources, and protects
recall: a company seen by only one system drops in rank but never
disappears.

Country is the only hard filter, applied after fusion, keeping the order.
Boolean filters from the plan (public status) work the same way. Numeric
thresholds get a tolerance band applied in code, not by the LLM: the model
extracts the number, the policy stays in the system.

The judge gets candidates in batches of 4 and has to return a verdict from a
closed set plus a verbatim quote as evidence for each company. The quote is
the main defense against made-up reasoning, and it doubles as material for
error analysis. If a batch fails validation or times out three times, it
gets marked insufficient_data with a distinctive reason instead of killing
the query.

LLM generation dominates the total latency. That's exactly why the funnel
exists: judgment reads 40 profiles, not 457, and that's what makes CPU-only
inference usable at all. The LLM client is an interface, so the same
pipeline runs unchanged against a hosted API when speed matters.

Metrics (precision@k, recall@k, nDCG@k with graded gains) are written from
scratch and cross-checked against the ranx library on a hand-computed case;
all four tests pass. Partial counts as relevant in the binary metrics,
consistent with the recall-first design. Evaluation code imports from the
system, never the other way around.

## 3.4 Results

All metrics run against the hand-verified golden set (167 pairs across q01,
q06, q12). Partial counts as relevant in binary metrics; nDCG uses graded
relevance (2/1/0). Four configurations, one stage added at a time:

| Query | Config          | P@10 | R@10 | nDCG@10 |
|-------|-----------------|------|------|---------|
| q01   | baseline        | 0.30 | 0.60 | 0.37    |
| q01   | hybrid+filter   | 0.20 | 0.40 | 0.17    |
| q01   | + rerank        | 0.30 | 0.60 | 0.40    |
| q01   | full pipeline   | 0.50 | 1.00 | 0.61    |
| q06   | baseline        | 1.00 | 0.30 | 1.00    |
| q06   | hybrid+filter   | 1.00 | 0.30 | 1.00    |
| q06   | + rerank        | 1.00 | 0.30 | 1.00    |
| q06   | full pipeline   | 1.00 | 0.30 | 1.00    |
| q12   | baseline        | 0.90 | 0.18 | 0.69    |
| q12   | hybrid+filter   | 0.90 | 0.18 | 0.79    |
| q12   | + rerank        | 0.90 | 0.18 | 0.78    |
| q12   | full pipeline   | 1.00 | 0.20 | 0.80    |

R@150 = 1.00 for every configuration and query: nothing relevant was ever
lost before the rerank boundary.

q01 tells the whole story. The baseline sees logistics vocabulary but not
roles. Fusion then hurts the top: the only fully relevant company (a postal
operator) is strong in dense search but invisible to BM25, because its
profile says "postal services, parcel delivery" and never "logistics", so
rank fusion drowns it. The cross-encoder gets it back by reading query and
profile together. Judgment then removes the having-vs-selling false
positives (a truck maker, a retailer with an internal logistics platform, an
oil company with its own supply chain): recall hits 1.00, precision 0.50.
Each stage fixes exactly the error class it was built for.

q06 never moves. The baseline is already perfect on this homogeneous query
and no later stage breaks it. R@10 = 0.30 is the arithmetic ceiling: 34
relevant companies don't fit in 10 positions. Queries like this could be
routed to skip the expensive judgment stage entirely.

q12 improves in ordering, not membership. RRF lifts nDCG from 0.69 to 0.79
by pushing material suppliers above cell makers; judgment cleans the top to
P@10 = 1.00. Low R@10 is structural again: 49 relevant companies exist. The
rerank cutoff at 40 caps recall on wide queries like this, a cost bound I
discuss in 3.5.

## 3.5 Failure modes and limits

BM25 on absent terms. For a term that appears nowhere in the corpus
(Shopify, q10), BM25 doesn't return an empty list: it returns k documents
with zero scores in index order. The absence of signal looks like a ranking.
Fusion dilutes the noise when the dense side has signal, but actually using
the absence (to declare a neutral verdict on unverifiable constraints) would
need score inspection, and rank-only fusion throws the scores away.

Fusion drowns single-source hits. RRF rewards consensus. The one fully
relevant q01 company is strong in dense, invisible in BM25, so fusion pushed
it out of the top 10. It stayed in the top 150 and the cross-encoder
recovered it. The lesson is in the table: stage 2 optimizes recall at its
boundary, judging it by precision at the top is the wrong metric.

Small-model enum escape. Twice, on genuinely ambiguous candidates, the 8B
judge invented verdicts outside the closed set: "match (partial)" and
"no_match, insufficient_data". The taxonomy itself was under pressure: the
golden set has a partial grade, the verdict schema doesn't. Fixed with a
strict format rule plus a legal exit (if torn, pick insufficient_data), and
a degradation path: a batch that fails three times gets marked
insufficient_data instead of killing the query.

Judgment inconsistency. On q01 the judge correctly rejected a truck maker, a
retailer and an oil company on the having-vs-selling rule, then approved two
other companies whose evidence quotes literally contain "own logistics
infrastructure". The rule exists but fires stochastically. Both errors are
visible in the evidence text itself, so a simple post-filter on evidence
wording is a cheap future fix.

Structural recall caps. The rerank cutoff at 40 bounds judgment cost but
also caps recall on wide queries: q12 has 49 relevant companies, so recall
can't pass 0.82 by construction. The cutoff should be per-query or adaptive;
with a fixed budget I documented the bound instead of silently missing it.

Verdict loss in batches. One run saved 39 verdicts for 40 candidates: the
model skipped a company and the response still validated, because a shorter
list is structurally legal. An unjudged company is silently excluded. A
count check per batch would catch this.

Taxonomy mismatch at evaluation. The golden set grades relevance (relevant /
partial / irrelevant), the judge emits states (match / no_match /
insufficient_data), and partial has no counterpart: the judge pushed
partially relevant ports and rail freight into match. Not a model error, a
design seam, and it's only visible because the evaluation has graded labels.