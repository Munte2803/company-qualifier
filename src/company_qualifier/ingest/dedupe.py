import json
from collections import Counter, defaultdict

from .models import Company


def _fingerprint(company: Company) -> str:
    data = company.model_dump()
    data.pop("row_index", None)
    return json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)


def dedupe_exact(companies: list[Company]) -> tuple[list[Company], Counter]:
    
    seen: dict[str, Company] = {}
    removed: Counter = Counter()

    for company in companies:
        key = _fingerprint(company)
        if key in seen:
            label = company.operational_name or company.website or f"row {company.row_index}"
            removed[label] += 1
            continue
        seen[key] = company

    return list(seen.values()), removed


def find_name_groups(companies: list[Company]) -> dict[str, list[Company]]:
   
    groups: dict[str, list[Company]] = defaultdict(list)
    for company in companies:
        if company.operational_name:
            groups[company.operational_name].append(company)
    return {name: items for name, items in groups.items() if len(items) > 1}