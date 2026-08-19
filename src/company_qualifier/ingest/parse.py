import ast
import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from .models import Company


def is_empty(value) -> bool:
    """Gol = None, string doar cu spații, sau listă/dict fără elemente."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return bool(isinstance(value, (list, dict)) and len(value) == 0)


def as_dict(value, field: str, failures: Counter) -> dict:
    """Normalizează un câmp compus la dict, indiferent dacă a venit ca dict
    sau ca repr Python cu ghilimele simple. Nu ridică excepții: la eșec
    returnează {} și incrementează contorul."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
            failures[f"{field}: not a dict"] += 1
        except (ValueError, SyntaxError):
            failures[f"{field}: literal_eval failed"] += 1
        return {}
    failures[f"{field}: unexpected type {type(value).__name__}"] += 1
    return {}


def load_companies(path: str | Path) -> tuple[list[Company], Counter]:
    """Citește JSONL, normalizează câmpurile compuse, validează.
    O linie care eșuează nu oprește procesul: se numără și se raportează."""
    failures: Counter = Counter()
    companies: list[Company] = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                failures["json decode failed"] += 1
                continue

            raw["address"] = as_dict(raw.get("address"), "address", failures)
            raw["primary_naics"] = as_dict(raw.get("primary_naics"), "primary_naics", failures)
            if raw.get("secondary_naics") is not None:
                raw["secondary_naics"] = as_dict(
                    raw["secondary_naics"], "secondary_naics", failures
                )
            raw["row_index"] = i

            try:
                companies.append(Company(**raw))
            except ValidationError as e:
                failures[f"validation: {e.errors()[0]['loc']}"] += 1

    return companies, failures