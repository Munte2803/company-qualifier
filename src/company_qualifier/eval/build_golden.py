

import csv
import random
import sys
from pathlib import Path

from company_qualifier.ingest.dedupe import dedupe_exact
from company_qualifier.ingest.parse import load_companies

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/companies.jsonl")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/golden/pool.tsv")
SEED = 42
N_RANDOM = 25

KW_Q12 = ["batter", "lithium", "cathode", "anode", "graphite", "electrolyte",
          "separator", "electric vehicle", " ev ", "nickel", "cobalt"]
NAICS_Q12 = ("3359", "32518")


def blob(c) -> str:
    return (c.description + " " + " ".join(c.core_offerings)
            + " " + " ".join(c.target_markets)).lower()


def build_pools(companies):
    pools = {}

    pools["q01"] = [(c, "country") for c in companies if c.address.country_code == "ro"]
    pools["q06"] = [(c, "country") for c in companies if c.address.country_code == "ch"]

    kw = {c.row_index for c in companies if any(k in blob(c) for k in KW_Q12)}
    naics = {c.row_index for c in companies if c.primary_naics.code.startswith(NAICS_Q12)}
    covered = kw | naics

    random.seed(SEED)
    rest = [c for c in companies if c.row_index not in covered]
    sampled = {c.row_index for c in random.sample(rest, min(N_RANDOM, len(rest)))}

    def source(i: int) -> str:
        parts = []
        if i in kw:
            parts.append("keyword")
        if i in naics:
            parts.append("naics")
        if i in sampled:
            parts.append("random")
        return "+".join(parts)

    pools["q12"] = [(c, source(c.row_index)) for c in companies
                    if c.row_index in covered | sampled]
    return pools


def write(pools, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["query_id", "row_index", "label", "source", "name", "country",
                    "naics_code", "naics_label", "description", "offerings"])
        for qid, pool in pools.items():
            for c, src in pool:
                w.writerow([qid, c.row_index, "", src,
                            c.operational_name or "(fără nume)",
                            c.address.country_code,
                            c.primary_naics.code, c.primary_naics.label,
                            c.description.replace("\t", " ").replace("\n", " "),
                            "; ".join(c.core_offerings)])


if __name__ == "__main__":
    companies, failures = load_companies(SRC)
    if failures:
        print("eșecuri la parsare:", dict(failures))

    companies, removed = dedupe_exact(companies)
    if removed:
        print("duplicate eliminate:", sum(removed.values()))

    pools = build_pools(companies)
    write(pools, OUT)
    print({k: len(v) for k, v in pools.items()},
          "→", sum(len(v) for v in pools.values()), "perechi")