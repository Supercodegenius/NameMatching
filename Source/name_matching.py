from __future__ import annotations

# Build with AI: AI-Powered Name Matching 
# Dashboards with Streamlit
# Name Matching Algorithms

# Developed By Ambuj Kumar

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Callable, Iterable, Literal

if TYPE_CHECKING:
    import pandas as pd

try:
    from rapidfuzz import distance as _rf_distance
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import process as _rf_process
except Exception:
    _rf_distance = None
    _rf_fuzz = None
    _rf_process = None

MatchMethod = Literal["exact", "fuzzy", "levenshtein", "jaro_winkler", "soundex", "ai_advanced"]
LevenshteinEngine = Literal["auto", "rapidfuzz", "python"]

def normalize_name(value: str) -> str:
    """Normalize names for more reliable comparisons."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def fuzzy_score(a: str, b: str) -> int:
    """Return similarity score in [0, 100]."""
    if _rf_fuzz is not None:
        return int(round(_rf_fuzz.ratio(a, b)))
    return int(round(100 * SequenceMatcher(None, a, b).ratio()))


def soundex_code(value: str) -> str:
    """Return 4-character Soundex code for a name."""
    letters = re.sub(r"[^a-z]", "", str(value).lower())
    if not letters:
        return ""

    first_letter = letters[0].upper()
    mapping = {
        "b": "1",
        "f": "1",
        "p": "1",
        "v": "1",
        "c": "2",
        "g": "2",
        "j": "2",
        "k": "2",
        "q": "2",
        "s": "2",
        "x": "2",
        "z": "2",
        "d": "3",
        "t": "3",
        "l": "4",
        "m": "5",
        "n": "5",
        "r": "6",
    }

    encoded_digits: list[str] = []
    prev_digit = mapping.get(letters[0], "")
    for ch in letters[1:]:
        digit = mapping.get(ch, "0")
        if digit != "0" and digit != prev_digit:
            encoded_digits.append(digit)
        prev_digit = digit

    code = first_letter + "".join(encoded_digits)
    return (code + "000")[:4]


def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    prev[j] + 1,  # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + cost,  # substitution
                )
            )
        prev = curr
    return prev[-1]


def levenshtein_distance_bounded(a: str, b: str, max_dist: int) -> int:
    """
    Compute Levenshtein distance with an upper bound.

    Returns max_dist + 1 when true distance is greater than max_dist.
    """
    if a == b:
        return 0
    if max_dist < 0:
        return max_dist + 1
    if not a:
        return len(b) if len(b) <= max_dist else max_dist + 1
    if not b:
        return len(a) if len(a) <= max_dist else max_dist + 1

    len_a = len(a)
    len_b = len(b)
    if abs(len_a - len_b) > max_dist:
        return max_dist + 1

    if len_a > len_b:
        a, b = b, a
        len_a, len_b = len_b, len_a

    inf = max_dist + 1

    prev_start = 0
    prev_end = min(len_b, max_dist)
    prev = list(range(prev_start, prev_end + 1))

    for i in range(1, len_a + 1):
        start = max(0, i - max_dist)
        end = min(len_b, i + max_dist)
        curr = [inf] * (end - start + 1)
        row_min = inf
        ai = a[i - 1]

        for j in range(start, end + 1):
            curr_idx = j - start

            if prev_start <= j <= prev_end:
                delete_cost = prev[j - prev_start] + 1
            else:
                delete_cost = inf

            if j == 0:
                insert_cost = i
            elif j - 1 >= start:
                insert_cost = curr[curr_idx - 1] + 1
            else:
                insert_cost = inf

            if prev_start <= j - 1 <= prev_end:
                substitute_cost = prev[j - 1 - prev_start] + (0 if ai == b[j - 1] else 1)
            else:
                substitute_cost = inf

            val = delete_cost if delete_cost < insert_cost else insert_cost
            if substitute_cost < val:
                val = substitute_cost

            curr[curr_idx] = val
            if val < row_min:
                row_min = val

        if row_min > max_dist:
            return inf

        prev = curr
        prev_start = start
        prev_end = end

    if prev_start <= len_b <= prev_end:
        result = prev[len_b - prev_start]
        return result if result <= max_dist else inf
    return inf


def jaro_similarity(a: str, b: str) -> float:
    """Compute Jaro similarity in [0.0, 1.0]."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    a_len = len(a)
    b_len = len(b)
    match_window = max(a_len, b_len) // 2 - 1
    if match_window < 0:
        match_window = 0

    a_matches = [False] * a_len
    b_matches = [False] * b_len

    matches = 0
    for i, ca in enumerate(a):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, b_len)
        for j in range(start, end):
            if b_matches[j]:
                continue
            if ca != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    a_match_chars: list[str] = [a[i] for i in range(a_len) if a_matches[i]]
    b_match_chars: list[str] = [b[j] for j in range(b_len) if b_matches[j]]

    transpositions = 0
    for ac, bc in zip(a_match_chars, b_match_chars):
        if ac != bc:
            transpositions += 1
    transpositions /= 2

    return (
        (matches / a_len)
        + (matches / b_len)
        + ((matches - transpositions) / matches)
    ) / 3.0


def jaro_winkler_similarity(a: str, b: str, *, prefix_scale: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity in [0.0, 1.0]."""
    jaro = jaro_similarity(a, b)
    if jaro == 0.0:
        return 0.0

    max_prefix = 4
    prefix = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        prefix += 1
        if prefix >= max_prefix:
            break

    jw = jaro + prefix * prefix_scale * (1.0 - jaro)
    if jw > 1.0:
        return 1.0
    if jw < 0.0:
        return 0.0
    return jw


def jaro_winkler_score(a: str, b: str) -> int:
    """Return Jaro-Winkler similarity score in [0, 100]."""
    if _rf_distance is not None:
        return int(round(100 * _rf_distance.JaroWinkler.similarity(a, b)))
    return int(round(100 * jaro_winkler_similarity(a, b)))


def token_set_score(a: str, b: str) -> int:
    """Return token-overlap score in [0, 100]."""
    left_tokens = set(a.split())
    right_tokens = set(b.split())
    if not left_tokens and not right_tokens:
        return 100
    if not left_tokens or not right_tokens:
        return 0
    overlap = len(left_tokens & right_tokens)
    total = len(left_tokens | right_tokens)
    return int(round(100 * (overlap / total)))


def _token_set_score_from_sets(left_tokens: set[str], right_tokens: set[str]) -> int:
    if not left_tokens and not right_tokens:
        return 100
    if not left_tokens or not right_tokens:
        return 0
    overlap = len(left_tokens & right_tokens)
    total = len(left_tokens | right_tokens)
    return int(round(100 * (overlap / total)))


def _first_token(text: str) -> str:
    return text.split(" ", 1)[0] if text else ""


def ai_advanced_score(a: str, b: str) -> int:
    """Weighted hybrid score designed for robust person-name matching."""
    fuzzy = fuzzy_score(a, b)
    jaro = jaro_winkler_score(a, b)
    token = token_set_score(a, b)

    weighted = (0.45 * jaro) + (0.35 * fuzzy) + (0.20 * token)

    left_first = a.split()[0] if a.split() else ""
    right_first = b.split()[0] if b.split() else ""
    if left_first and left_first == right_first:
        weighted += 5

    weighted = max(0.0, min(100.0, weighted))
    return int(round(weighted))


def _build_candidate_getter(target_normalized: list[str]) -> Callable[[str], list[int]]:
    """
    Return a fast candidate lookup function.

    For small target lists we keep full-scan behavior to preserve exact ranking.
    For larger lists we use lightweight blocking and bounded fallbacks.
    """
    target_count = len(target_normalized)
    if target_count <= 3000:
        all_indices = list(range(target_count))
        return lambda _src_n: all_indices

    first_char_index: dict[str, list[int]] = defaultdict(list)
    prefix_index: dict[str, list[int]] = defaultdict(list)
    token_index: dict[str, list[int]] = defaultdict(list)
    length_index: dict[int, list[int]] = defaultdict(list)

    for idx, tgt_n in enumerate(target_normalized):
        length_index[len(tgt_n)].append(idx)
        if not tgt_n:
            continue
        first_char_index[tgt_n[0]].append(idx)
        prefix_index[tgt_n[:3]].append(idx)
        for token in {tok for tok in tgt_n.split() if len(tok) >= 2}:
            token_index[token].append(idx)

    all_indices = list(range(target_count))

    max_candidates = 400
    max_pool = 800
    max_bucket_take = 200

    def get_candidates(src_n: str) -> list[int]:
        if not src_n:
            return all_indices

        seen: set[int] = set()
        ordered: list[int] = []

        def add_indices(indices: list[int], limit: int | None = None) -> None:
            if not indices:
                return
            added = 0
            for idx in indices:
                if idx in seen:
                    continue
                seen.add(idx)
                ordered.append(idx)
                added += 1
                if len(ordered) >= max_pool:
                    return
                if limit is not None and added >= limit:
                    return

        add_indices(prefix_index.get(src_n[:3], []))
        add_indices(first_char_index.get(src_n[:1], []))

        src_tokens = [tok for tok in src_n.split() if len(tok) >= 2]
        src_tokens.sort(key=len, reverse=True)
        for token in src_tokens[:4]:
            add_indices(token_index.get(token, []), max_bucket_take)
            if len(ordered) >= max_pool:
                break

        src_len = len(src_n)
        for delta in range(0, 3):
            near_lengths = [src_len - delta] if delta == 0 else [src_len - delta, src_len + delta]
            for length_val in near_lengths:
                if length_val < 0:
                    continue
                add_indices(length_index.get(length_val, []), max_bucket_take)
                if len(ordered) >= max_pool:
                    break
            if len(ordered) >= max_pool:
                break

        if not ordered:
            return all_indices

        if len(ordered) > max_candidates:
            return ordered[:max_candidates]
        return ordered

    return get_candidates


def match_names(
    source_names: Iterable[str],
    target_names: Iterable[str],
    *,
    method: MatchMethod,
    fuzzy_threshold: int = 75,
    lev_max_distance: int = 2,
    lev_engine: LevenshteinEngine = "auto",
) -> pd.DataFrame:
    """Return match results for each source name against a target list."""
    import pandas as pd

    src_series = pd.Series(list(source_names)).fillna("").astype(str)
    tgt_series = pd.Series(list(target_names)).fillna("").astype(str)

    src_norm = src_series.map(normalize_name)
    tgt_norm = tgt_series.map(normalize_name)

    right_lookup = pd.DataFrame(
        {"target_original": tgt_series, "target_normalized": tgt_norm}
    ).drop_duplicates(subset=["target_normalized"], keep="first")

    results: list[dict] = []

    if method == "exact":
        right_map = dict(
            zip(right_lookup["target_normalized"], right_lookup["target_original"])
        )
        for src, src_n in zip(src_series, src_norm):
            matched = right_map.get(src_n)
            results.append(
                {
                    "source_name": src,
                    "source_normalized": src_n,
                    "matched_name": matched if matched is not None else "",
                    "score": 100 if matched is not None else 0,
                    "is_match": matched is not None,
                }
            )
        return pd.DataFrame(results)

    target_originals = right_lookup["target_original"].tolist()
    target_normalized = right_lookup["target_normalized"].tolist()
    target_exact_map = dict(zip(target_normalized, target_originals))
    get_candidates = _build_candidate_getter(target_normalized)

    if method == "fuzzy":
        best_cache: dict[str, dict] = {}
        for src, src_n in zip(src_series, src_norm):
            if src_n not in best_cache:
                candidate_indices = get_candidates(src_n)
                if not candidate_indices:
                    candidate_indices = list(range(len(target_originals)))

                best_name = ""
                best_score = 0
                for idx in candidate_indices:
                    score = fuzzy_score(src_n, target_normalized[idx])
                    if score > best_score:
                        best_score = score
                        best_name = target_originals[idx]
                best_cache[src_n] = {
                    "source_normalized": src_n,
                    "matched_name": best_name,
                    "score": best_score,
                    "is_match": best_score >= fuzzy_threshold,
                }

            results.append({"source_name": src, **best_cache[src_n]})
        return pd.DataFrame(results)

    if method == "soundex":
        soundex_lookup = right_lookup.copy()
        soundex_lookup["target_soundex"] = soundex_lookup["target_normalized"].map(soundex_code)
        right_map = dict(
            zip(soundex_lookup["target_soundex"], soundex_lookup["target_original"])
        )
        for src, src_n in zip(src_series, src_norm):
            src_soundex = soundex_code(src_n)
            matched = right_map.get(src_soundex)
            results.append(
                {
                    "source_name": src,
                    "source_normalized": src_n,
                    "source_soundex": src_soundex,
                    "matched_name": matched if matched is not None else "",
                    "score": 100 if matched is not None else 0,
                    "is_match": matched is not None,
                }
            )
        return pd.DataFrame(results)

    if method == "jaro_winkler":
        best_cache = {}
        for src, src_n in zip(src_series, src_norm):
            if src_n not in best_cache:
                candidate_indices = get_candidates(src_n)
                if not candidate_indices:
                    candidate_indices = list(range(len(target_originals)))

                best_name = ""
                best_score = 0
                for idx in candidate_indices:
                    score = jaro_winkler_score(src_n, target_normalized[idx])
                    if score > best_score:
                        best_score = score
                        best_name = target_originals[idx]
                best_cache[src_n] = {
                    "source_normalized": src_n,
                    "matched_name": best_name,
                    "score": best_score,
                    "is_match": best_score >= fuzzy_threshold,
                }

            results.append({"source_name": src, **best_cache[src_n]})
        return pd.DataFrame(results)

    if method == "levenshtein":
        best_cache = {}
        all_indices = list(range(len(target_originals)))
        use_rapidfuzz = (
            lev_engine == "auto" and _rf_process is not None and _rf_distance is not None
        ) or (lev_engine == "rapidfuzz" and _rf_process is not None and _rf_distance is not None)
        for src, src_n in zip(src_series, src_norm):
            if src_n not in best_cache:
                exact_match = target_exact_map.get(src_n)
                if exact_match is not None:
                    best_cache[src_n] = {
                        "source_normalized": src_n,
                        "matched_name": exact_match,
                        "distance": 0,
                        "score": 100,
                        "is_match": True,
                    }
                    results.append({"source_name": src, **best_cache[src_n]})
                    continue

                best_name = ""
                best_distance = 10**9
                best_target_norm = ""

                candidate_indices = get_candidates(src_n)
                if not candidate_indices:
                    candidate_indices = all_indices
                elif len(candidate_indices) <= 256:
                    # For small candidate sets, sort by length to tighten bounds quickly.
                    src_len = len(src_n)
                    candidate_indices = sorted(
                        candidate_indices,
                        key=lambda idx: abs(len(target_normalized[idx]) - src_len),
                    )

                if use_rapidfuzz:
                    candidate_names = [target_normalized[idx] for idx in candidate_indices]
                    best_hit = _rf_process.extractOne(
                        src_n,
                        candidate_names,
                        scorer=_rf_distance.Levenshtein.distance,
                        processor=None,
                    )
                    if best_hit is not None:
                        best_target_norm = best_hit[0]
                        best_distance = int(best_hit[1])
                        best_name = target_originals[candidate_indices[int(best_hit[2])]]
                else:
                    for idx in candidate_indices:
                        tgt_n = target_normalized[idx]
                        if best_distance != 10**9 and abs(len(src_n) - len(tgt_n)) >= best_distance:
                            continue

                        if best_distance == 10**9:
                            dist = levenshtein_distance(src_n, tgt_n)
                        else:
                            dist = levenshtein_distance_bounded(src_n, tgt_n, best_distance - 1)
                            if dist >= best_distance:
                                continue

                        if dist < best_distance:
                            best_distance = dist
                            best_name = target_originals[idx]
                            best_target_norm = tgt_n
                            if best_distance == 0:
                                break

                max_len = max(len(src_n), len(best_target_norm), 1)
                similarity_score = int(round(100 * (1 - (best_distance / max_len))))
                best_cache[src_n] = {
                    "source_normalized": src_n,
                    "matched_name": best_name,
                    "distance": best_distance,
                    "score": similarity_score,
                    "is_match": best_distance <= lev_max_distance,
                }

            results.append({"source_name": src, **best_cache[src_n]})
        return pd.DataFrame(results)

    if method == "ai_advanced":
        best_cache = {}
        target_first_tokens = [_first_token(name) for name in target_normalized]
        target_token_sets = [set(name.split()) for name in target_normalized]
        for src, src_n in zip(src_series, src_norm):
            if src_n not in best_cache:
                candidate_indices = get_candidates(src_n)
                if not candidate_indices:
                    candidate_indices = list(range(len(target_originals)))

                src_first_token = _first_token(src_n)
                src_token_set = set(src_n.split())
                best_name = ""
                best_score = 0
                best_fuzzy = 0
                best_jaro = 0
                best_token = 0
                for idx in candidate_indices:
                    tgt_n = target_normalized[idx]
                    fuzzy = fuzzy_score(src_n, tgt_n)
                    jaro = jaro_winkler_score(src_n, tgt_n)
                    token = _token_set_score_from_sets(src_token_set, target_token_sets[idx])
                    score = int(
                        round(
                            max(
                                0.0,
                                min(
                                    100.0,
                                    (0.45 * jaro)
                                    + (0.35 * fuzzy)
                                    + (0.20 * token)
                                    + (5 if (src_first_token and src_first_token == target_first_tokens[idx]) else 0),
                                ),
                            )
                        )
                    )
                    if score > best_score:
                        best_score = score
                        best_name = target_originals[idx]
                        best_fuzzy = fuzzy
                        best_jaro = jaro
                        best_token = token
                best_cache[src_n] = {
                    "source_normalized": src_n,
                    "matched_name": best_name,
                    "score": best_score,
                    "ai_fuzzy_score": best_fuzzy,
                    "ai_jaro_winkler_score": best_jaro,
                    "ai_token_score": best_token,
                    "is_match": best_score >= fuzzy_threshold,
                }

            results.append({"source_name": src, **best_cache[src_n]})
        return pd.DataFrame(results)

    raise ValueError(f"Unknown method: {method}")
