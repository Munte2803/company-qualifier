
from collections.abc import Callable

from .models import Company
from .parse import is_empty

EXTRACTORS: dict[str, Callable[[Company], str]] = {
    "name": lambda c: c.operational_name or "",
    "naics": lambda c: c.primary_naics.label,
    "description": lambda c: c.description,
    "offerings": lambda c: ", ".join(c.core_offerings),
    "markets": lambda c: ", ".join(c.target_markets),
    "business_model": lambda c: ", ".join(c.business_model),
    "country": lambda c: c.address.country_code,
}

PREFIXES: dict[str, str] = {
    "offerings": "offerings: ",
    "markets": "markets: ",
    "business_model": "model: ",
    "country": "country: ",
}

DEFAULT_FIELDS: tuple[str, ...] = (
    "name",
    "naics",
    "description",
    "offerings",
    "markets",
)


def build_profile(
    company: Company,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> str:

    parts: list[str] = []

    for field in fields:
        extractor = EXTRACTORS.get(field)
        if extractor is None:
            raise KeyError(f"câmp necunoscut în profil: {field!r}")

        value = extractor(company)
        if is_empty(value):
            continue

        parts.append(PREFIXES.get(field, "") + value.strip().rstrip("."))

    return ". ".join(parts) + "." if parts else ""


def build_all(
    companies: list[Company],
    fields: tuple[str, ...] = DEFAULT_FIELDS,
) -> list[str]:
    return [build_profile(c, fields) for c in companies]