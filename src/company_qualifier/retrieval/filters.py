def filter_country(ranking:list[int], country_code: str, countries: dict[int, str]) -> list[int]:
     """Returnează companiile care au țara specificată."""
     return [c for c in ranking if countries.get(c) == country_code]