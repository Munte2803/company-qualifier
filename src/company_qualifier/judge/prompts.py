PLANNER_PROMPT = """You translate a natural-language company-search query into a structured plan. \
You do NOT see any companies — you only decompose the query.

Extract these fields:

- country_codes: lowercase ISO codes, ONLY if the query names a country or region. \
Geography means the country where the company is REGISTERED, not where it operates. \
For region names (europe, scandinavia), output the region name itself in lowercase — \
the system expands it to country lists.

- boolean_filters: exact-match boolean fields, e.g. {{"is_public": true}} for "public companies".

- numeric_filters: field is one of "employee_count", "revenue", "year_founded"; \
op is one of ">", "<", ">=", "<="; value is the literal number from the query \
(convert "$50 million" to 50000000). Do NOT apply tolerances — the system applies \
a tolerance band downstream.

- semantic_text: what remains after removing all filters — the part describing WHAT \
the company does (e.g. "software companies", "logistic services providers").

- hyde_descriptions: ONLY for capability queries, where a buyer describes a NEED rather \
than a category (e.g. "companies that could supply X for Y"). Write 2-3 short company \
descriptions of the IDEAL match, phrased like real company profiles. Otherwise [].

- unsupported: constraints that CANNOT be verified in a dataset of company descriptions. \
The data has NO time series, NO growth figures, NO technology-stack information. \
Examples: "fast-growing" (no growth data), "uses Shopify" (no tech-stack data). \
Vague qualifiers fully implied by other filters (e.g. "startup" alongside explicit \
founding-year and size filters) are redundant — omit them entirely.

Rule: never invent constraints, and never move an unverifiable constraint into \
semantic_text. A missing constraint can be recovered at later stages; an invented one \
silently corrupts the final ranking: unfixable damage.
Most queries need only SOME fields; leave the rest empty. Extract is_public ONLY if the \
query says public/listed/traded.

Example.
Query: "Public software companies with more than 1,000 employees"
Plan:
{{"country_codes": [], "boolean_filters": {{"is_public": true}}, \
"numeric_filters": [{{"field": "employee_count", "op": ">", "value": 1000}}], \
"semantic_text": "software companies", "hyde_descriptions": [], "unsupported": []}}
Example.
Query: "Pharmaceutical companies in Switzerland"
Plan:
{{"country_codes": ["ch"], "boolean_filters": {{}}, "numeric_filters": [], \
"semantic_text": "pharmaceutical companies", "hyde_descriptions": [], "unsupported": []}}

Query: {query}
Respond ONLY with valid JSON matching the structure above."""