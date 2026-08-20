def filter_country(ranking:list[int], country_code: str, countries: dict[int, str]) -> list[int]:
     """Returnează companiile care au țara specificată."""
     return [c for c in ranking if countries.get(c) == country_code]
def filter_boolean(ranking: list[int], flags: dict[str, bool],
                   
                   values: dict[int, dict[str, bool]]) -> list[int]:
    if not flags:
        return ranking
    return [i for i in ranking
            if all(values.get(i, {}).get(f) == v for f, v in flags.items())]