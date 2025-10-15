#!/usr/bin/env python3
"""
Lottery analysis for 新澳门六合彩 (Macau Mark Six) to recommend next 特码 (special code).

- Input: CSV file with at least 288 draws (ideally 2025年001期-288期). Columns can be flexible:
  - Period/Issue: e.g., period, 期数, 期號, issue, draw_no, no
  - 6 正码 columns: e.g., z1..z6, 正1..正6, number1..number6
  - 1 特码 column: e.g., tema, 特码, 特码, special, tm
  - Optionally a single column containing all numbers, e.g., "numbers" like "01,02,03,04,05,06+07"

- Output: Console summary with hot/cold numbers and a recommendation for the next 特码 (第289期),
  plus alternates. Optionally JSON output via --out-json.

NOTE: This is statistical/heuristic analysis for entertainment only; it cannot predict outcomes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class Draw:
    period: Optional[int]
    zheng: List[int]
    te: int
    year: Optional[int] = None


def _to_int_safe(val: str) -> Optional[int]:
    try:
        v = int(str(val).strip())
        return v
    except Exception:
        # try removing non-digits
        digits = "".join(ch for ch in str(val) if ch.isdigit())
        try:
            return int(digits) if digits else None
        except Exception:
            return None


def _parse_numbers_blob(blob: str) -> Optional[Tuple[List[int], int]]:
    """
    Parse a single-field representation like "01 02 03 04 05 06 + 07" or "1,2,3,4,5,6,7".
    Assumes the last number is 特码.
    """
    if blob is None:
        return None
    s = str(blob)
    # normalize separators
    for sep in ["+", "|", "-", "/", "\\", "；", "|", "·", "·"]:
        s = s.replace(sep, ",")
    s = s.replace(" ", ",").replace("；", ",").replace("、", ",").replace("，", ",")
    parts = [p for p in s.split(",") if p.strip()]
    nums: List[int] = []
    for p in parts:
        n = _to_int_safe(p)
        if n is None:
            continue
        nums.append(n)
    nums = [n for n in nums if 1 <= n <= 49]
    if len(nums) < 7:
        return None
    # Take last as 特码
    return nums[:6], nums[6]


def _detect_columns(headers: List[str]) -> Dict[str, List[int]]:
    """Detect likely column indices for period, six 正码, and 特码.
    Returns mapping: { 'period': [idx?], 'zheng': [idx...], 'te': [idx?], 'year': [idx?] }
    """
    lowered = [h.lower() for h in headers]

    period_aliases = {"period", "issue", "draw", "no", "期数", "期號", "期号", "期"}
    te_aliases = {"tema", "te", "tm", "special", "sp", "特碼", "特码", "特码", "特"}
    year_aliases = {"year", "年份"}

    # Positive signals for 正码 columns
    zheng_signals = ["正", "zheng", "z", "z1", "z2", "z3", "z4", "z5", "z6", "num", "number"]

    col_map: Dict[str, List[int]] = {"period": [], "zheng": [], "te": [], "year": []}

    for idx, h in enumerate(headers):
        h_l = lowered[idx]
        if any(alias in h_l for alias in period_aliases):
            col_map["period"].append(idx)
        if any(alias in h_l for alias in te_aliases):
            col_map["te"].append(idx)
        if any(alias in h_l for alias in year_aliases):
            col_map["year"].append(idx)
        if any(sig in h_l for sig in zheng_signals):
            col_map["zheng"].append(idx)

    return col_map


def _row_to_draw(row: Dict[str, str], headers: List[str]) -> Optional[Draw]:
    # Try direct detection from columns
    col_map = _detect_columns(headers)

    # Single blob fallback
    for key in ["numbers", "nums", "号码", "開獎號碼", "开奖号码", "開獎", "開奬", "開獎球", "开奖号码"]:
        if key in row and row[key]:
            parsed = _parse_numbers_blob(row[key])
            if parsed:
                zheng, te = parsed
                period = None
                # Guess period if present in other fields
                for pkey in ["period", "期数", "期號", "期号", "issue", "draw", "no"]:
                    if pkey in row:
                        p = _to_int_safe(row[pkey])
                        period = p if p is not None else period
                # Guess year
                year = None
                for ykey in ["year", "年份", "date", "日期"]:
                    if ykey in row:
                        y = _to_int_safe(str(row[ykey])[:4])
                        if y and 1900 < y < 2100:
                            year = y
                return Draw(period=period, zheng=zheng[:6], te=te, year=year)

    # Otherwise, collect numeric-like fields
    values: List[Tuple[int, str]] = []
    for h in headers:
        v = row.get(h)
        n = _to_int_safe(v)
        if n is not None and 0 <= n <= 9999:  # consider period/years as well
            values.append((n, h))

    # Identify 特码 value
    te_val: Optional[int] = None
    te_indices = _detect_columns(headers)["te"]
    if te_indices:
        for idx in te_indices:
            v = row.get(headers[idx])
            n = _to_int_safe(v)
            if n is not None and 1 <= n <= 49:
                te_val = n
                break
    if te_val is None:
        # Heuristic: among 1..49 numbers, if we have >=7, use the last one as 特码
        numbered = [n for n, _ in values if 1 <= n <= 49]
        if len(numbered) >= 7:
            te_val = numbered[6]

    # Collect 正码 candidates
    zheng_vals: List[int] = []
    z_indices = _detect_columns(headers)["zheng"]
    if z_indices:
        for idx in z_indices:
            n = _to_int_safe(row.get(headers[idx]))
            if n is not None and 1 <= n <= 49:
                zheng_vals.append(n)
    else:
        # Fallback: take first 6 1..49 numbers
        for n, _ in values:
            if 1 <= n <= 49 and len(zheng_vals) < 6:
                zheng_vals.append(n)

    if len(zheng_vals) < 6 or te_val is None:
        return None

    # Period and year
    period: Optional[int] = None
    for key in ["period", "期数", "期號", "期号", "issue", "draw", "no"]:
        if key in row:
            p = _to_int_safe(row[key])
            period = p if p is not None else period

    year: Optional[int] = None
    for key in ["year", "年份", "date", "日期"]:
        if key in row:
            y_guess = _to_int_safe(str(row[key])[:4])
            if y_guess and 1900 < y_guess < 2100:
                year = y_guess

    # Deduplicate 正码 in the unlikely event of repeated values
    zheng_vals = [n for i, n in enumerate(zheng_vals) if n not in zheng_vals[:i]]
    if len(zheng_vals) > 6:
        zheng_vals = zheng_vals[:6]

    return Draw(period=period, zheng=zheng_vals, te=te_val, year=year)


def load_draws(csv_path: str) -> List[Draw]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        # sniff dialect
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        headers = reader.fieldnames or []
        if not headers:
            raise ValueError("CSV has no header row")

        draws: List[Draw] = []
        for row in reader:
            d = _row_to_draw(row, headers)
            if d is not None:
                draws.append(d)

    # Filter numbers strictly within 1..49 and 6 zheng + 1 te
    cleaned: List[Draw] = []
    for d in draws:
        z = [n for n in d.zheng if 1 <= n <= 49]
        if len(z) != 6:
            continue
        if not (1 <= d.te <= 49):
            continue
        cleaned.append(Draw(period=d.period, zheng=z, te=d.te, year=d.year))

    return cleaned


@dataclass
class FrequencyResult:
    freq_overall: Counter
    freq_zheng: Counter
    freq_te: Counter
    decayed_overall: Dict[int, float]
    decayed_zheng: Dict[int, float]
    decayed_te: Dict[int, float]


def compute_frequencies(draws: Sequence[Draw], half_life: float = 60.0) -> FrequencyResult:
    freq_overall: Counter = Counter()
    freq_zheng: Counter = Counter()
    freq_te: Counter = Counter()

    for d in draws:
        freq_te[d.te] += 1
        for n in d.zheng:
            freq_zheng[n] += 1
            freq_overall[n] += 1
        freq_overall[d.te] += 1

    # Exponential decay for recency (newest draw has index len(draws)-1)
    # weight = 0.5 ** (age / half_life)
    decayed_overall: Dict[int, float] = defaultdict(float)
    decayed_zheng: Dict[int, float] = defaultdict(float)
    decayed_te: Dict[int, float] = defaultdict(float)

    total = len(draws)
    for idx, d in enumerate(draws):
        age = total - 1 - idx
        weight = 0.5 ** (age / max(1.0, half_life))
        for n in d.zheng:
            decayed_zheng[n] += weight
            decayed_overall[n] += weight
        decayed_te[d.te] += weight
        decayed_overall[d.te] += weight

    return FrequencyResult(
        freq_overall=freq_overall,
        freq_zheng=freq_zheng,
        freq_te=freq_te,
        decayed_overall=dict(decayed_overall),
        decayed_zheng=dict(decayed_zheng),
        decayed_te=dict(decayed_te),
    )


def _quantile_threshold(counts: Dict[int, float], q: float) -> float:
    values = sorted(counts.get(n, 0.0) for n in range(1, 50))
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, int(math.floor(q * (len(values) - 1)))))
    return values[idx]


def _normalize(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _bounds(counts: Dict[int, float]) -> Tuple[float, float]:
    vals = [counts.get(n, 0.0) for n in range(1, 50)]
    return (min(vals), max(vals)) if vals else (0.0, 1.0)


def analyze_alternation(
    draws: Sequence[Draw],
    freq: FrequencyResult,
    hot_q: float = 0.72,  # top 28% considered hot
    cold_q: float = 0.28,  # bottom 28% considered cold
    lookback: int = 60,
) -> Dict[str, object]:
    # Thresholds for hot/cold in 正码 and 特码 perspectives
    th_hot_zheng = _quantile_threshold(freq.freq_zheng, hot_q)
    th_cold_zheng = _quantile_threshold(freq.freq_zheng, cold_q)
    th_hot_te = _quantile_threshold(freq.freq_te, hot_q)
    th_cold_te = _quantile_threshold(freq.freq_te, cold_q)

    # Classify each draw's 特码 relative to 正码/特码 distributions
    # classes: 'HZ_CT' (hot-in-zheng, cold-in-te), 'CZ_HT' (cold-in-zheng, hot-in-te), 'OTHER'
    classes: List[str] = []
    for d in draws:
        te = d.te
        fz = freq.freq_zheng.get(te, 0)
        ft = freq.freq_te.get(te, 0)
        is_hz = fz >= th_hot_zheng
        is_cz = fz <= th_cold_zheng
        is_ht = ft >= th_hot_te
        is_ct = ft <= th_cold_te
        cls = "OTHER"
        if is_hz and is_ct:
            cls = "HZ_CT"
        elif is_cz and is_ht:
            cls = "CZ_HT"
        classes.append(cls)

    window = classes[-lookback:] if lookback > 0 else classes
    alt_count = 0
    trans_count = 0
    for a, b in zip(window, window[1:]):
        if a in ("HZ_CT", "CZ_HT") and b in ("HZ_CT", "CZ_HT"):
            trans_count += 1
            if a != b:
                alt_count += 1

    alternation_rate = (alt_count / trans_count) if trans_count else 0.0

    last_cls = classes[-1] if classes else "OTHER"
    # Predict next class: if alternation tendency > 0.5, flip the last class; else keep last if meaningful, else choose the more frequent class
    if last_cls == "HZ_CT":
        predicted = "CZ_HT" if alternation_rate > 0.5 else "HZ_CT"
    elif last_cls == "CZ_HT":
        predicted = "HZ_CT" if alternation_rate > 0.5 else "CZ_HT"
    else:
        hz_ct = window.count("HZ_CT")
        cz_ht = window.count("CZ_HT")
        predicted = "HZ_CT" if hz_ct >= cz_ht else "CZ_HT"

    return {
        "predicted_class": predicted,
        "alternation_rate": alternation_rate,
        "last_class": last_cls,
        "thresholds": {
            "hot_zheng": th_hot_zheng,
            "cold_zheng": th_cold_zheng,
            "hot_te": th_hot_te,
            "cold_te": th_cold_te,
        },
    }


def rank_candidates(
    freq: FrequencyResult,
    prediction: Dict[str, object],
    widen_if_few: int = 8,
) -> List[Tuple[int, float, Dict[str, float]]]:
    # Build hot/cold sets based on thresholds
    th = prediction["thresholds"]  # type: ignore
    th_hot_zheng = float(th["hot_zheng"])  # type: ignore
    th_cold_zheng = float(th["cold_zheng"])  # type: ignore
    th_hot_te = float(th["hot_te"])  # type: ignore
    th_cold_te = float(th["cold_te"])  # type: ignore

    hot_zheng = {n for n in range(1, 50) if freq.freq_zheng.get(n, 0) >= th_hot_zheng}
    cold_zheng = {n for n in range(1, 50) if freq.freq_zheng.get(n, 0) <= th_cold_zheng}
    hot_te = {n for n in range(1, 50) if freq.freq_te.get(n, 0) >= th_hot_te}
    cold_te = {n for n in range(1, 50) if freq.freq_te.get(n, 0) <= th_cold_te}

    predicted_class = str(prediction["predicted_class"])  # type: ignore
    if predicted_class == "HZ_CT":
        base_candidates = (hot_zheng & cold_te)
    else:  # CZ_HT
        base_candidates = (cold_zheng & hot_te)

    # Widen if too few candidates
    widen_steps = 0
    while len(base_candidates) < widen_if_few and widen_steps < 3:
        widen_steps += 1
        # Widen thresholds progressively: include more numbers near thresholds
        # We approximate widening by mixing in next-most candidates by score proxy
        if predicted_class == "HZ_CT":
            # include numbers that are near-hot zheng (top 50%) and near-cold te (bottom 50%)
            q_hot_zh = _quantile_threshold(freq.freq_zheng, 0.5)
            q_cold_te = _quantile_threshold(freq.freq_te, 0.5)
            extra = {n for n in range(1, 50) if (freq.freq_zheng.get(n, 0) >= q_hot_zh and freq.freq_te.get(n, 0) <= q_cold_te)}
            base_candidates |= extra
        else:
            q_cold_zh = _quantile_threshold(freq.freq_zheng, 0.5)
            q_hot_te = _quantile_threshold(freq.freq_te, 0.5)
            extra = {n for n in range(1, 50) if (freq.freq_zheng.get(n, 0) <= q_cold_zh and freq.freq_te.get(n, 0) >= q_hot_te)}
            base_candidates |= extra

    # Normalize ranges for scoring
    lo_ov, hi_ov = _bounds(freq.freq_overall)
    lo_z, hi_z = _bounds(freq.freq_zheng)
    lo_t, hi_t = _bounds(freq.freq_te)
    lo_do, hi_do = _bounds(freq.decayed_overall)
    lo_dz, hi_dz = _bounds(freq.decayed_zheng)
    lo_dt, hi_dt = _bounds(freq.decayed_te)

    def score_number(n: int) -> Tuple[float, Dict[str, float]]:
        fz = freq.freq_zheng.get(n, 0)
        ft = freq.freq_te.get(n, 0)
        fo = freq.freq_overall.get(n, 0)
        dz = freq.decayed_zheng.get(n, 0.0)
        dt = freq.decayed_te.get(n, 0.0)
        do = freq.decayed_overall.get(n, 0.0)

        nfz = _normalize(fz, lo_z, hi_z)
        nft = _normalize(ft, lo_t, hi_t)
        nfo = _normalize(fo, lo_ov, hi_ov)
        ndz = _normalize(dz, lo_dz, hi_dz)
        ndt = _normalize(dt, lo_dt, hi_dt)
        ndo = _normalize(do, lo_do, hi_do)

        # Repeat bias: if it was recently 特码, allow slight repetition (+), else 0
        repeat_bias = 0.1 * ndt

        if predicted_class == "HZ_CT":
            s = 0.60 * nfz - 0.30 * nft + 0.25 * ndz - 0.15 * ndt + 0.15 * nfo + 0.10 * ndo + repeat_bias
        else:  # CZ_HT
            s = 0.60 * nft - 0.30 * nfz + 0.25 * ndt - 0.15 * ndz + 0.15 * nfo + 0.10 * ndo + repeat_bias

        # Deterministic tiny jitter to break ties (function of number)
        jitter = (n % 7) * 1e-6
        return s + jitter, {
            "nfz": nfz, "nft": nft, "nfo": nfo, "ndz": ndz, "ndt": ndt, "ndo": ndo,
        }

    scored: List[Tuple[int, float, Dict[str, float]]] = []
    for n in sorted(base_candidates):
        s, parts = score_number(n)
        scored.append((n, s, parts))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def select_last_288_2025(draws: List[Draw]) -> List[Draw]:
    # Filter for year 2025 if available; else take last 288 overall
    draws_2025 = [d for d in draws if d.year == 2025]
    selected = draws_2025 if len(draws_2025) >= 288 else draws
    if len(selected) < 288:
        # best-effort: still proceed with what we have
        return selected[-min(288, len(selected)) :]
    return selected[:288]


def summarize_hot_cold(freq: FrequencyResult, top_n: int = 10) -> Dict[str, List[int]]:
    # Top/bottom by counts
    def top_k(counter: Counter, k: int) -> List[int]:
        return [n for n, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]

    def bottom_k(counter: Counter, k: int) -> List[int]:
        # Include numbers absent in the counter too
        items = [(n, counter.get(n, 0)) for n in range(1, 50)]
        items.sort(key=lambda kv: (kv[1], kv[0]))
        return [n for n, _ in items[:k]]

    return {
        "hot_overall": top_k(freq.freq_overall, top_n),
        "cold_overall": bottom_k(freq.freq_overall, top_n),
        "hot_zheng": top_k(freq.freq_zheng, top_n),
        "cold_zheng": bottom_k(freq.freq_zheng, top_n),
        "hot_te": top_k(freq.freq_te, top_n),
        "cold_te": bottom_k(freq.freq_te, top_n),
    }


def recommend(
    draws: List[Draw],
    n_alternates: int = 5,
) -> Dict[str, object]:
    if not draws:
        raise ValueError("No draws available for analysis")

    freq = compute_frequencies(draws)
    alt = analyze_alternation(draws, freq)
    candidates = rank_candidates(freq, alt)

    if not candidates:
        # Fallback: choose top by 特码 frequency
        fallback = sorted(freq.freq_te.items(), key=lambda kv: (-kv[1], kv[0]))
        best = fallback[0][0] if fallback else 1
        alts = [n for n, _ in fallback[1 : n_alternates + 1]]
        return {
            "recommendation": best,
            "alternates": alts,
            "predicted_class": alt["predicted_class"],
            "alternation_rate": alt["alternation_rate"],
            "notes": "Fallback to 特码 frequency due to empty candidate set",
        }

    best = candidates[0][0]
    alts = [n for n, _, _ in candidates[1 : n_alternates + 1]]

    hot_cold_summary = summarize_hot_cold(freq, top_n=10)

    return {
        "recommendation": best,
        "alternates": alts,
        "predicted_class": alt["predicted_class"],
        "alternation_rate": alt["alternation_rate"],
        "hot_cold": hot_cold_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 新澳门六合彩 draws and recommend next 特码 (entertainment only)")
    parser.add_argument("--csv", required=True, help="Path to CSV with at least 288 draws")
    parser.add_argument("--year", type=int, default=2025, help="Filter to this year if present (default: 2025)")
    parser.add_argument("--out-json", default=None, help="Optional path to write JSON output")
    parser.add_argument("--alternates", type=int, default=5, help="Number of alternate candidates to output")
    args = parser.parse_args()

    draws = load_draws(args.csv)

    # Select 2025 draws if available, else last 288 overall
    if args.year:
        picks = [d for d in draws if d.year == args.year]
        draws_sel = picks if len(picks) >= 288 else draws
    else:
        draws_sel = draws
    if len(draws_sel) >= 288:
        # Prefer first 288 draws of the selected period range
        draws_sel = draws_sel[:288]
    else:
        # Take as many as available up to 288
        draws_sel = draws_sel[-min(288, len(draws_sel)) :]

    result = recommend(draws_sel, n_alternates=args.alternates)

    # Human-readable summary
    print("=== 新澳门六合彩 正码+特码交替 分析 (近288期) ===")
    print(f"样本期数: {len(draws_sel)} 期")
    print(f"交替趋势(近窗): {result.get('alternation_rate'):.3f}  预测类别: {result.get('predicted_class')}")
    hot_cold = result.get("hot_cold") or {}
    if hot_cold:
        print("\n热门号码(正码):", hot_cold.get("hot_zheng"))
        print("冷门号码(正码):", hot_cold.get("cold_zheng"))
        print("热门号码(特码):", hot_cold.get("hot_te"))
        print("冷门号码(特码):", hot_cold.get("cold_te"))
    print("\n【第289期 特码 推荐】:", result.get("recommendation"))
    print("备选:", result.get("alternates"))
    print("\n提示: 仅为数据分析参考与娱乐，不构成任何保证或建议。")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 已输出: {args.out_json}")


if __name__ == "__main__":
    main()
