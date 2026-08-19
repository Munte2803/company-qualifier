from company_qualifier.judge.llm import OllamaClient
from company_qualifier.judge.schemas import JudgeResponse, Verdict

JUDGE_PROMPT = """You are qualifying companies against a search query. For EACH company below, \
decide independently:

- "match": the profile explicitly supports ALL requirements of the query
- "no_match": the profile contradicts at least one requirement
- "insufficient_data": the profile neither confirms nor denies — do NOT guess

Rules: evidence must be a SHORT VERBATIM quote from that company's profile. \
Having something internally (e.g. own logistics) is NOT the same as selling it as a service. \
Judge each company on its own profile only.

Query: {query}

Companies:
{companies}

Respond ONLY with valid JSON: {{"verdicts": [{{"row_index": <int>, "verdict": "...", \
"evidence": "...", "reason": "..."}}]}} — one entry per company, in the given order."""


def judge_candidates(client: OllamaClient, query: str,
                     candidates: list[int], profiles: dict[int, str],
                     batch_size: int = 4) -> list[Verdict]:
    verdicts: list[Verdict] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        block = "\n\n".join(f"[row_index={i}]\n{profiles[i]}" for i in batch)
        resp = client.complete(JUDGE_PROMPT.format(query=query, companies=block),
                               JudgeResponse)
        verdicts.extend(resp.verdicts)
        print(f"  judged {start + len(batch)}/{len(candidates)}")
    return verdicts