

LEGAL_SUFFIXES = [
    "ltd", "limited", "plc", "llc", "inc", "corp", "co", "company",
    "ag", "sa", "s.a.", "s.a", "spa", "gmbh", "bv", "nv", "oy", "ab",
    "pte", "pvt", "kg", "kgaa", "sas", "sarl", "llp", "lp"
]
REINSURANCE_TERMS = [
    "reinsurance", "reinsurance company", "reinsurance co",
    "re", "re.", "reins", "reins.", "reassurance",
    "underwriting", "insurance", "insurance company",
    "branch", "uk branch", "europe sa", "europe s.a."
]

import unicodedata
import re

def clean_text(s):
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def remove_terms(s, terms):
    tokens = s.split()
    filtered = [t for t in tokens if t not in terms]
    return " ".join(filtered)

def normalise_name(name):
    s = clean_text(name)

    s = remove_terms(s, LEGAL_SUFFIXES)
    s = remove_terms(s, REINSURANCE_TERMS)

    # Remove standalone country codes
    COUNTRY_TERMS = ["uk", "usa", "us", "eu", "europa", "asia", "emea"]
    s = remove_terms(s, COUNTRY_TERMS)

    # Remove repeated spaces again
    s = " ".join(s.split())

    return s
