# -*- coding: utf-8 -*-
"""
Macau Mark Six (新澳门六合彩) 2025 analysis and prediction pipeline.
- Scrape 2025 (期数001-292) from https://kj.123720c.com/kj/?year=2025
- Parse zodiac and attribute tables from https://kj.123720c.com/kj/sx.html?year=2025 and https://kj.123720c.com/kj/zl.html
- Compute stats: frequency, omission, trend (EMA), co-occurrence matrix
- Strategies: frequency-balanced, omission, trend, association, clustering, simple-ML ranking
- Backtest over full 001-292 and rolling windows [292, 120, 80, 50, 15]
- Output CSVs and a chat-ready textual report

Notes:
- Designed to run without heavy ML dependencies; uses numpy/pandas only
- Robust HTML parsing with BeautifulSoup if available; falls back to regex
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "amlhc", "raw")
OUT_DIR = os.path.join("data", "amlhc", "out")

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux) AppleWebKit/537.36 Chrome/118 Safari/537.36"}


@dataclass
class Draw:
    issue: int  # 1..292 within year
    date: str   # e.g., 2025年10月19日
    nums: List[int]  # 6 normal numbers (平码)
    tm: int  # 特码
    colors: List[str]  # ['red'|'green'|'blue'] per each number in nums + tm? stored for 7 length
    zodiacs: List[str]  # zodiac per number (7 length)

    def all7(self) -> List[int]:
        return self.nums + [self.tm]


@dataclass
class Attributes2025:
    # For 2025 (蛇年), mapping number->attributes
    num_to_zodiac: Dict[int, str]
    num_to_color: Dict[int, str]
    num_to_five_element: Dict[int, str]
    num_to_odd_even: Dict[int, str]
    num_to_size: Dict[int, str]
    zodiac_to_poultry_beast: Dict[str, str]  # 生肖->家禽/野兽


def ensure_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)


def http_get(url: str, save_path: Optional[str] = None, force: bool = False) -> str:
    if save_path and (not force) and os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if requests is None:
        raise RuntimeError("requests is not available. Install dependencies.")
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    text = r.text
    if save_path:
        with open(save_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(text)
    return text


def parse_draws_2025(html: str) -> List[Draw]:
    # Each period block structure:
    # <div class="kj-tit">六合彩开奖记录 YYYY年MM月DD日 第<span class="text-blue text-strong">NNN</span>期</div>
    # <div class="kj-box"><ul class="clearfix"> ... 7 li's with one plus separator 'li class="kj-jia"' ...</ul></div>

    # Split by titles to pair with subsequent kj-box
    title_iter = list(re.finditer(
        r"<div class=\"kj-tit\">([^<]*?)第<span class=\\\"text-blue text-strong\\\">(\\d+)<\\/span>期<\\/div>",
        html
    ))
    # Fallback: less escaped
    if not title_iter:
        title_iter = list(re.finditer(
            r"<div class=\"kj-tit\">([^<]*?)第<span class=\"text-blue text-strong\">(\d+)</span>期</div>",
            html
        ))
    draws: List[Draw] = []
    for m in title_iter:
        date_text = m.group(1).strip()
        issue = int(m.group(2))
        # Find the nearest kj-box after this position
        start = m.end()
        box_m = re.search(r"<div class=\"kj-box\">(.*?)</div>", html[start:], flags=re.S)
        if not box_m:
            continue
        box_html = box_m.group(1)
        # Extract all number items in order; each item is in <dt class="ball-...">NN</dt>
        items = re.findall(r"<dt class=\"ball-(red|green|blue)\">\s*(\d{1,2})\s*</dt>\s*<dd>([^<]+)<font", box_html)
        # items length is 7
        if len(items) < 7:
            # try more robust parse via BeautifulSoup if available
            nums_seq: List[Tuple[str, int, str]] = []
            if BeautifulSoup is not None:
                soup = BeautifulSoup(box_html, "html.parser")
                for li in soup.find_all("li"):
                    dt = li.find("dt")
                    dd = li.find("dd")
                    if dt and dd and (dt.text.strip().isdigit()):
                        cls = ""
                        for c in (dt.get("class") or []):
                            if c.startswith("ball-"):
                                cls = c.split("-", 1)[1]
                        nums_seq.append((cls, int(dt.text.strip()), dd.text.strip()[:1]))
            if len(nums_seq) >= 7:
                colors = [c for c, n, z in nums_seq[:7]]
                numbers = [n for c, n, z in nums_seq[:7]]
                zodiacs = [z for c, n, z in nums_seq[:7]]
            else:
                # regex fallback: pick any digits in dt and zodiac single char before '/'
                seq = re.findall(r"<dt class=\"ball-(red|green|blue)\">\s*(\d{1,2})\s*</dt>\s*<dd>\s*([^<\s])[\s\S]*?</dd>", box_html)
                if len(seq) >= 7:
                    colors = [c for c, n, z in seq[:7]]
                    numbers = [int(n) for c, n, z in seq[:7]]
                    zodiacs = [z for c, n, z in seq[:7]]
                else:
                    continue
        else:
            colors = [c for c, n, z in items[:7]]
            numbers = [int(n) for c, n, z in items[:7]]
            zodiacs = [z[:1] for c, n, z in items[:7]]
        nums = numbers[:6]
        tm = numbers[6]
        draws.append(Draw(issue=issue, date=date_text.strip(), nums=nums, tm=tm, colors=colors, zodiacs=zodiacs))
    # Ensure sorted by issue ascending (1..292)
    draws.sort(key=lambda d: d.issue)
    return draws


def parse_attributes_2025(sx_html: str, zl_html: str) -> Attributes2025:
    # Zodiac mapping lines like: <span class="bg_green">06 </span> under 十二生肖 listing per zodiac order
    # The page layout groups zodiacs with images and numbers; extract blocks between <span class="sx">生肖</span> and next <li>
    zodiac_order = [
        "鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"
    ]
    # Build mapping num->zodiac by scanning occurrences of zx header then following numbers until next zodiac header
    num_to_zodiac: Dict[int, str] = {}
    if BeautifulSoup is not None:
        soup = BeautifulSoup(sx_html, "html.parser")
        ul = soup.find("ul", class_="sxdz")
        if not ul:
            ul = soup
        current_zx: Optional[str] = None
        for elem in ul.descendants:
            if getattr(elem, "name", None) == "span" and "sx" in (elem.get("class") or []):
                # text like '鼠[冲 马]'
                t = elem.get_text(strip=True)
                m = re.match(r"([鼠牛虎兔龙蛇马羊猴鸡狗猪])", t)
                if m:
                    current_zx = m.group(1)
            elif getattr(elem, "name", None) == "span" and re.search(r"bg_(red|green|blue)", " ".join(elem.get("class") or [])):
                num_txt = elem.get_text(" ", strip=True)
                mnum = re.search(r"(\d{1,2})", num_txt)
                if mnum and current_zx:
                    num = int(mnum.group(1))
                    num_to_zodiac[num] = current_zx
    # Fallback: regex by sections under 十二生肖
    if not num_to_zodiac:
        for zx in zodiac_order:
            pattern = rf"{zx}.*?(?:</span>|</strong>)([\s\S]*?)(?:<span class=\"sx\"|</li>)"
            for m in re.finditer(pattern, sx_html):
                section = m.group(1)
                for mnum in re.finditer(r"(\d{1,2})", section):
                    num_to_zodiac[int(mnum.group(1))] = zx
    # Color mapping from 波色表 in sx_html under 红波/蓝波/绿波
    num_to_color: Dict[int, str] = {}
    for color_name, key in [("红波", "red"), ("蓝波", "blue"), ("绿波", "green")]:
        sec = re.search(rf">{color_name}<.*?</td>\s*<td>([\s\S]*?)</td>", sx_html)
        if sec:
            for n in re.findall(r"(\d{1,2})", sec.group(1)):
                num_to_color[int(n)] = key
    # Five elements mapping from table 金/木/水/火/土
    num_to_five_element: Dict[int, str] = {}
    for element in ["金", "木", "水", "火", "土"]:
        sec = re.search(rf">{element}<.*?</td>\s*([\s\S]*?)</tr>", sx_html)
        if sec:
            for n in re.findall(r"(\d{1,2})", sec.group(1)):
                num_to_five_element[int(n)] = element
    # Odd/Even and Size from standard definitions
    num_to_odd_even = {n: ("单" if n % 2 == 1 else "双") for n in range(1, 50)}
    num_to_size = {n: ("大" if n >= 25 else "小") for n in range(1, 50)}
    # Poultry/Beast mapping from domain knowledge; also appears in result hidden dd as 家禽/野兽 pattern
    poultry = set(["牛", "马", "羊", "鸡", "狗", "猪"])  # 家禽
    beast = set(["鼠", "虎", "兔", "龙", "蛇", "猴"])     # 野兽
    zodiac_to_poultry_beast = {zx: ("家禽" if zx in poultry else "野兽") for zx in zodiac_order}
    return Attributes2025(
        num_to_zodiac=num_to_zodiac,
        num_to_color=num_to_color,
        num_to_five_element=num_to_five_element,
        num_to_odd_even=num_to_odd_even,
        num_to_size=num_to_size,
        zodiac_to_poultry_beast=zodiac_to_poultry_beast,
    )


def draws_to_dataframe(draws: List[Draw]) -> pd.DataFrame:
    rows = []
    for d in draws:
        row: Dict[str, Any] = {"issue": d.issue, "date": d.date}
        for i, n in enumerate(d.nums, start=1):
            row[f"p{i}"] = n
        row["tm"] = d.tm
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("issue").reset_index(drop=True)
    return df


def enrich_with_attributes(df: pd.DataFrame, attrs: Attributes2025) -> pd.DataFrame:
    def z(x: int) -> str:
        return attrs.num_to_zodiac.get(int(x), "")

    def c(x: int) -> str:
        return attrs.num_to_color.get(int(x), "")

    def oe(x: int) -> str:
        return attrs.num_to_odd_even.get(int(x), "")

    def sz(x: int) -> str:
        return attrs.num_to_size.get(int(x), "")

    def fe(x: int) -> str:
        return attrs.num_to_five_element.get(int(x), "")

    for col in ["p1", "p2", "p3", "p4", "p5", "p6", "tm"]:
        df[f"{col}_zodiac"] = df[col].map(z)
        df[f"{col}_color"] = df[col].map(c)
        df[f"{col}_odd_even"] = df[col].map(oe)
        df[f"{col}_size"] = df[col].map(sz)
        df[f"{col}_five_element"] = df[col].map(fe)
        df[f"{col}_poultry_beast"] = df[f"{col}_zodiac"].map(lambda zx: attrs.zodiac_to_poultry_beast.get(zx, ""))
    return df


def compute_frequencies(df: pd.DataFrame, window: int) -> Counter:
    tail = df.tail(window) if window < len(df) else df
    nums = tail[["p1", "p2", "p3", "p4", "p5", "p6", "tm"]].values.reshape(-1)
    return Counter(int(x) for x in nums)


def compute_last_seen_omission(df: pd.DataFrame) -> Dict[int, int]:
    last_seen: Dict[int, int] = {n: -1 for n in range(1, 50)}
    for _, row in df.iterrows():
        issue = int(row["issue"])
        for col in ["p1", "p2", "p3", "p4", "p5", "p6", "tm"]:
            n = int(row[col])
            last_seen[n] = issue
    max_issue = int(df["issue"].max())
    omission = {n: (max_issue - last_seen[n]) if last_seen[n] != -1 else max_issue for n in range(1, 50)}
    return omission


def compute_trend_ema(df: pd.DataFrame, span: int = 20) -> Dict[int, float]:
    # Build time-series indicator for each number
    issues = df["issue"].astype(int).tolist()
    max_issue = issues[-1]
    ema_scores = {n: 0.0 for n in range(1, 50)}
    alpha = 2.0 / (span + 1)
    # Initialize with 0, update sequentially
    for _, row in df.iterrows():
        present = set(int(x) for x in [row["p1"], row["p2"], row["p3"], row["p4"], row["p5"], row["p6"], row["tm"]])
        for n in range(1, 50):
            x = 1.0 if n in present else 0.0
            ema_scores[n] = alpha * x + (1 - alpha) * ema_scores[n]
    return ema_scores


def compute_cooccurrence(df: pd.DataFrame, window: Optional[int] = None) -> np.ndarray:
    tail = df.tail(window) if window and window < len(df) else df
    mat = np.zeros((50, 50), dtype=np.int32)
    for _, row in tail.iterrows():
        s = set(int(x) for x in [row["p1"], row["p2"], row["p3"], row["p4"], row["p5"], row["p6"], row["tm"]])
        for a in s:
            for b in s:
                if a != b:
                    mat[a, b] += 1
    return mat


def cluster_numbers_by_cooccur(mat: np.ndarray, k: int = 6) -> Dict[int, List[int]]:
    # Simple k-means on rows 1..49 with cosine distance
    rng = np.random.default_rng(42)
    X = mat[1:50, 1:50].astype(np.float64)
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    Xn = X / norms
    # Init centers as random rows
    idxs = rng.choice(Xn.shape[0], size=k, replace=False)
    centers = Xn[idxs, :]
    for _ in range(20):
        # Assign
        sims = Xn @ centers.T  # cosine similarity
        labels = np.argmax(sims, axis=1)
        # Update
        for j in range(k):
            members = Xn[labels == j]
            if len(members) > 0:
                centers[j] = members.mean(axis=0)
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(Xn.shape[0]):
        clusters[int(labels[i])].append(i + 1)  # number = index+1
    return clusters


def simple_ml_ranking(df: pd.DataFrame, windows: List[int]) -> Dict[int, float]:
    # Build features per number per draw: recent frequencies in provided windows and recency (omission)
    # Train linear weights by least squares to approximate next-draw presence (one-step-ahead)
    numbers = list(range(1, 50))
    feats_list: List[List[float]] = []
    y_list: List[float] = []
    # We will create samples for draws t from 50..end-1, predicting presence in draw t+1
    for t in range(50, len(df) - 1):
        past = df.iloc[: t + 1]
        nxt = df.iloc[t + 1]
        # compute features at time t
        freq_feats: List[float] = []
        for w in windows:
            cnt = compute_frequencies(past, min(w, len(past)))
            freq_feats.extend([cnt.get(n, 0) / max(1, min(w, len(past))) for n in numbers])
        # omission at time t
        omission = compute_last_seen_omission(past)
        om_feats = [omission[n] / 100.0 for n in numbers]
        # Combine
        feats = freq_feats + om_feats
        feats_list.append(feats)
        present_next = set(int(x) for x in [nxt["p1"], nxt["p2"], nxt["p3"], nxt["p4"], nxt["p5"], nxt["p6"], nxt["tm"]])
        y = np.array([1.0 if n in present_next else 0.0 for n in numbers], dtype=np.float64)
        y_list.append(y.mean())  # average presence rate across numbers (sparse); acts as global target
    if not feats_list:
        return {n: 0.0 for n in numbers}
    X = np.array(feats_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    # Ridge-like solution with small L2
    XtX = X.T @ X + 1e-3 * np.eye(X.shape[1])
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)
    # Current features
    past = df
    freq_feats_cur: List[float] = []
    for w in windows:
        cnt = compute_frequencies(past, min(w, len(past)))
        freq_feats_cur.extend([cnt.get(n, 0) / max(1, min(w, len(past))) for n in numbers])
    omission = compute_last_seen_omission(past)
    om_feats_cur = [omission[n] / 100.0 for n in numbers]
    feats_cur = np.array(freq_feats_cur + om_feats_cur, dtype=np.float64)
    # Score per-number via projecting features onto w equally for each number block proportionally
    # Since w predicts global rate, distribute by standardized per-number feature importance using latest window frequency and recency
    recent_cnt = compute_frequencies(df, min(50, len(df)))
    base = np.array([recent_cnt.get(n, 0) for n in numbers], dtype=np.float64)
    base = (base - base.mean()) / (base.std() + 1e-9)
    om = np.array([omission[n] for n in numbers], dtype=np.float64)
    om = (om - om.mean()) / (om.std() + 1e-9)
    scores = 0.6 * base + 0.4 * om
    # Normalize to 0..1
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return {numbers[i]: float(scores[i]) for i in range(len(numbers))}


def pick_balanced_by_color_odd_even(candidates: List[int], attrs: Attributes2025, k: int = 7) -> List[int]:
    # Ensure diversity across colors and odd/even
    buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for n in candidates:
        buckets[(attrs.num_to_color.get(n, ""), "单" if n % 2 == 1 else "双")].append(n)
    picks: List[int] = []
    # Round-robin from buckets sorted by availability
    bucket_keys = sorted(buckets.keys(), key=lambda b: -len(buckets[b]))
    while len(picks) < k and any(buckets.values()):
        for bk in bucket_keys:
            if buckets[bk]:
                x = buckets[bk].pop(0)
                if x not in picks:
                    picks.append(x)
                if len(picks) >= k:
                    break
    # Fill if needed
    for n in candidates:
        if len(picks) >= k:
            break
        if n not in picks:
            picks.append(n)
    return picks[:k]


def strategy_predictions(df: pd.DataFrame, attrs: Attributes2025) -> Dict[str, Dict[str, Any]]:
    windows = [292, 120, 80, 50, 15]
    full_freq = compute_frequencies(df, len(df))
    omissions = compute_last_seen_omission(df)
    ema = compute_trend_ema(df, span=20)
    co_mat = compute_cooccurrence(df)
    clusters = cluster_numbers_by_cooccur(co_mat, k=6)
    ml_scores = simple_ml_ranking(df, windows=[120, 50, 15])

    # Helpers
    def top_nums(counter: Counter, n: int) -> List[int]:
        return [num for num, _ in counter.most_common(n)]

    # Frequency-balanced: take top by frequency in 120 window, balance by color/odd-even
    freq120 = compute_frequencies(df, 120)
    freq_candidates = top_nums(freq120, 20)
    freq_balanced = pick_balanced_by_color_odd_even(freq_candidates, attrs, k=7)

    # Omission: take largest omissions
    omission_sorted = sorted(omissions.items(), key=lambda x: (-x[1], x[0]))
    omission_top = [n for n, _ in omission_sorted[:20]]
    omission_picks = pick_balanced_by_color_odd_even(omission_top, attrs, k=7)

    # Trend: top EMA
    trend_sorted = sorted(ema.items(), key=lambda x: (-x[1], x[0]))
    trend_top = [n for n, _ in trend_sorted[:20]]
    trend_picks = pick_balanced_by_color_odd_even(trend_top, attrs, k=7)

    # Association: with the last draw's set, pick numbers with highest co-occurrence
    last = df.iloc[-1]
    last_set = set(int(last[c]) for c in ["p1", "p2", "p3", "p4", "p5", "p6", "tm"])
    assoc_scores: Dict[int, int] = {}
    for n in range(1, 50):
        if n in last_set:
            continue
        assoc_scores[n] = int(sum(co_mat[n, m] for m in last_set))
    assoc_sorted = sorted(assoc_scores.items(), key=lambda x: (-x[1], x[0]))
    assoc_top = [n for n, _ in assoc_sorted[:20]]
    assoc_picks = pick_balanced_by_color_odd_even(assoc_top, attrs, k=7)

    # Clustering: pick one high-frequency number from each cluster
    cluster_picks: List[int] = []
    for cid, members in clusters.items():
        members_freq = sorted([(m, full_freq.get(m, 0)) for m in members], key=lambda x: (-x[1], x[0]))
        if members_freq:
            cluster_picks.append(members_freq[0][0])
    cluster_picks = sorted(cluster_picks, key=lambda n: (-full_freq.get(n, 0), n))[:7]

    # Simple ML: top by score
    ml_sorted = sorted(ml_scores.items(), key=lambda x: (-x[1], x[0]))
    ml_picks = [n for n, _ in ml_sorted[:7]]

    # choose a Tm candidate for each strategy: highest EMA among picks
    def pick_tm(picks: List[int]) -> int:
        return sorted(picks, key=lambda n: (-ema.get(n, 0.0), omissions.get(n, 0)), reverse=False)[0]

    def zx(n: int) -> str:
        return attrs.num_to_zodiac.get(n, "")

    out: Dict[str, Dict[str, Any]] = {
        "频率平衡预测": {"picks": freq_balanced, "tm": pick_tm(freq_balanced), "tm_zodiac": zx(pick_tm(freq_balanced))},
        "遗漏值预测": {"picks": omission_picks, "tm": pick_tm(omission_picks), "tm_zodiac": zx(pick_tm(omission_picks))},
        "趋势预测": {"picks": trend_picks, "tm": pick_tm(trend_picks), "tm_zodiac": zx(pick_tm(trend_picks))},
        "关联性预测": {"picks": assoc_picks, "tm": pick_tm(assoc_picks), "tm_zodiac": zx(pick_tm(assoc_picks))},
        "聚类预测": {"picks": cluster_picks, "tm": pick_tm(cluster_picks), "tm_zodiac": zx(pick_tm(cluster_picks))},
        "机器学习预测": {"picks": ml_picks, "tm": pick_tm(ml_picks), "tm_zodiac": zx(pick_tm(ml_picks))},
    }
    return out


def backtest_hits(df: pd.DataFrame, picks: List[int], window: Optional[int] = None) -> Dict[str, Any]:
    tail = df.tail(window) if window and window < len(df) else df
    hits_per_draw: List[int] = []
    for _, row in tail.iterrows():
        s = set(int(x) for x in [row["p1"], row["p2"], row["p3"], row["p4"], row["p5"], row["p6"], row["tm"]])
        hits_per_draw.append(len(s.intersection(picks)))
    return {
        "avg_hits": float(np.mean(hits_per_draw)) if hits_per_draw else 0.0,
        "max_hits": int(np.max(hits_per_draw)) if hits_per_draw else 0,
        "hit_rate_ge1": float(np.mean([1 if h >= 1 else 0 for h in hits_per_draw])) if hits_per_draw else 0.0,
        "n_draws": int(len(hits_per_draw)),
    }


def build_chat_report(strategies: Dict[str, Dict[str, Any]], df: pd.DataFrame, attrs: Attributes2025) -> str:
    # Hot/Cold numbers over 001-292
    freq_full = compute_frequencies(df, len(df))
    hot = [n for n, _ in freq_full.most_common(10)]
    cold = [n for n, _ in sorted(freq_full.items(), key=lambda x: (x[1], x[0]))[:10]]
    windows = [292, 120, 80, 50, 15]
    lines: List[str] = []
    lines.append("【2025年 新澳门六合彩 001-292期 热门/冷门号码】")
    lines.append(f"热门: {hot}")
    lines.append(f"冷门: {cold}")
    lines.append("")
    for w in windows:
        freqw = compute_frequencies(df, w)
        hotw = [n for n, _ in freqw.most_common(7)]
        lines.append(f"窗口{w}期高频: {hotw}")
    lines.append("")
    # Strategies with backtests
    for name, res in strategies.items():
        picks = res["picks"]
        tm = res["tm"]
        tm_zx = res["tm_zodiac"]
        bt_full = backtest_hits(df, picks, window=None)
        bt_120 = backtest_hits(df, picks, window=120)
        bt_50 = backtest_hits(df, picks, window=50)
        lines.append(f"【{name}】")
        lines.append(f"推荐7码: {picks}；特码候选: {tm}({tm_zx})")
        lines.append(
            f"回测(全/120/50) 平均命中: {bt_full['avg_hits']:.2f}/{bt_120['avg_hits']:.2f}/{bt_50['avg_hits']:.2f}, "
            f"≥1命中率: {bt_full['hit_rate_ge1']:.2f}/{bt_120['hit_rate_ge1']:.2f}/{bt_50['hit_rate_ge1']:.2f}"
        )
        lines.append("")
    return "\n".join(lines)


def run_pipeline(fetch: bool, force: bool) -> str:
    ensure_dirs()
    # Fetch pages
    year_url = "https://kj.123720c.com/kj/?year=2025"
    sx_url = "https://kj.123720c.com/kj/sx.html?year=2025"
    zl_url = "https://kj.123720c.com/kj/zl.html"
    year_path = os.path.join(RAW_DIR, "kj_2025.html")
    sx_path = os.path.join(RAW_DIR, "kj_sx.html")
    zl_path = os.path.join(RAW_DIR, "kj_zl.html")

    if fetch:
        _ = http_get(year_url, year_path, force=force)
        _ = http_get(sx_url, sx_path, force=force)
        _ = http_get(zl_url, zl_path, force=force)
    else:
        # Ensure files exist
        for p in [year_path, sx_path, zl_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required file: {p}. Run with --fetch.")

    with open(year_path, "r", encoding="utf-8", errors="ignore") as f:
        year_html = f.read()
    with open(sx_path, "r", encoding="utf-8", errors="ignore") as f:
        sx_html = f.read()
    with open(zl_path, "r", encoding="utf-8", errors="ignore") as f:
        zl_html = f.read()

    draws = parse_draws_2025(year_html)
    if not draws or len(draws) < 292:
        # Allow slightly less if the last day hasn't been published; but target is 292
        pass

    df = draws_to_dataframe(draws)
    attrs = parse_attributes_2025(sx_html, zl_html)
    df_attr = enrich_with_attributes(df, attrs)

    # Save CSVs
    df.to_csv(os.path.join(OUT_DIR, "amlhc_2025_draws.csv"), index=False)
    df_attr.to_csv(os.path.join(OUT_DIR, "amlhc_2025_draws_attrs.csv"), index=False)

    # Stats and strategies
    strategies = strategy_predictions(df, attrs)
    with open(os.path.join(OUT_DIR, "amlhc_2025_strategies.json"), "w", encoding="utf-8") as f:
        json.dump(strategies, f, ensure_ascii=False, indent=2)

    report = build_chat_report(strategies, df, attrs)
    with open(os.path.join(OUT_DIR, "amlhc_2025_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Macau Mark Six 2025 analysis pipeline")
    parser.add_argument("--fetch", action="store_true", help="Fetch pages from network")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()
    report = run_pipeline(fetch=args.fetch, force=args.force)
    print(report)


if __name__ == "__main__":
    main()
