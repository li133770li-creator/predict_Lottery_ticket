# -*- coding: utf-8 -*-
"""
Analyze 新澳门六合彩 历史 001-285 期（页面当前年度内容）以提取正码/特码，
计算热门/冷门与交替指标，并给出 286 期 特码预测。

数据来源：已下载的本地文件 data/kj_index.html （来源首页 /kj/ 列表）。
"""
import re
import os
import sys
import math
import json
import statistics
from collections import Counter, defaultdict, deque
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import csv
from bs4 import BeautifulSoup

DATA_HTML_PATH = os.path.join("data", "kj_index.html")
# Optional extra pages for extended history
DATA_HTML_EXTRA = [
    os.path.join("data", "kj_2024.html"),
    os.path.join("data", "kj_2023.html"),
]
PARSED_CSV_PATH = os.path.join("data", "parsed_draws.csv")
REPORT_PATH = os.path.join("data", "analysis_report_001_285.md")
REPORT_500_PATH = os.path.join("data", "analysis_report_last500.md")

# Configurable parameters
NUMBERS_RANGE = range(1, 50)  # Mark Six numbers 1..49
HOT_COUNT = 10                 # Top N -> 热门
COLD_COUNT = 10                # Bottom N -> 冷门
ALT_WINDOW = 50                # 交替判断的滑窗长度
RECENT_WINDOW_FOR_FREQ = 100  # 预测时使用的最近窗口（频率与间隔评估）


def _safe_int(text: str) -> Optional[int]:
    try:
        return int(text)
    except Exception:
        return None


def parse_draws_from_html(html_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"未找到文件: {html_path}")

    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    result: List[Dict[str, Any]] = []

    # 每条记录由一个标题 div.kj-tit 和紧随的 div.kj-box 组成
    for tit_div in soup.find_all("div", class_="kj-tit"):
        # 示例: "六合彩开奖记录 2025年10月12日 第285期"
        tit_text = tit_div.get_text(" ", strip=True)

        # 抓取期号
        m_issue = re.search(r"第\s*(\d+)\s*期", tit_text)
        if not m_issue:
            continue
        issue = int(m_issue.group(1))

        # 抓取日期（可选）
        m_date = re.search(r"(\d{4}年\d{2}月\d{2}日)", tit_text)
        date_str = m_date.group(1) if m_date else None

        box_div = tit_div.find_next_sibling("div", class_="kj-box")
        if not box_div:
            continue

        ul = box_div.find("ul")
        if not ul:
            continue

        zms: List[int] = []
        tm: Optional[int] = None

        seen_plus = False
        for li in ul.find_all("li", recursive=False):
            li_classes = li.get("class") or []
            if "kj-jia" in li_classes:
                seen_plus = True
                continue

            dt = li.find("dt")
            if not dt:
                continue
            n = _safe_int(dt.get_text(strip=True))
            if n is None:
                continue

            if not seen_plus:
                zms.append(n)
            else:
                # 第一个加号后的球视为特码
                if tm is None:
                    tm = n

        # 过滤异常记录
        if len(zms) == 6 and tm is not None:
            result.append({
                "issue": issue,
                "date": date_str,
                "zms": zms,
                "tm": tm,
            })

    # 按期号升序，便于按时间序列分析
    result.sort(key=lambda x: x["issue"])
    return result


def parse_draws_from_multiple(html_paths: List[str]) -> List[Dict[str, Any]]:
    """Parse multiple HTML pages with the same structure and merge.
    Deduplicate by (issue, date, zms, tm) if overlapping.
    """
    merged: List[Dict[str, Any]] = []
    for p in html_paths:
        if p and os.path.exists(p):
            merged.extend(parse_draws_from_html(p))
    # sort and deduplicate by issue
    merged.sort(key=lambda x: (x["issue"], x.get("date") or ""))
    dedup: Dict[int, Dict[str, Any]] = {}
    for d in merged:
        dedup[d["issue"]] = d
    out = list(dedup.values())
    out.sort(key=lambda x: x["issue"])
    return out


def save_draws_to_csv(draws: List[Dict[str, Any]], csv_path: str) -> None:
    fieldnames = ["issue", "date", "zm1", "zm2", "zm3", "zm4", "zm5", "zm6", "tm"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in draws:
            row = {
                "issue": d["issue"],
                "date": d.get("date"),
                **{f"zm{i+1}": d["zms"][i] for i in range(6)},
                "tm": d["tm"],
            }
            writer.writerow(row)


def compute_frequencies(draws: List[Dict[str, Any]]) -> Dict[str, Counter]:
    zm_counter: Counter[int] = Counter()
    tm_counter: Counter[int] = Counter()
    all_counter: Counter[int] = Counter()

    for d in draws:
        for n in d["zms"]:
            zm_counter[n] += 1
            all_counter[n] += 1
        tm_counter[d["tm"]] += 1
        all_counter[d["tm"]] += 1

    # 确保 1..49 都有键，便于冷门识别
    for n in NUMBERS_RANGE:
        _ = zm_counter[n]
        _ = tm_counter[n]
        _ = all_counter[n]

    return {"zm": zm_counter, "tm": tm_counter, "all": all_counter}


def top_n(counter: Counter, n: int) -> List[Tuple[int, int]]:
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def bottom_n(counter: Counter, n: int) -> List[Tuple[int, int]]:
    return sorted(counter.items(), key=lambda kv: (kv[1], kv[0]))[:n]


def compute_hot_cold_sets(counter: Counter, hot_k: int, cold_k: int) -> Tuple[set, set]:
    hot = set([num for num, _ in top_n(counter, hot_k)])
    cold = set([num for num, _ in bottom_n(counter, cold_k)])
    return hot, cold


def windowed_frequency(draws: List[Dict[str, Any]], start_idx: int, end_idx: int) -> Counter:
    """闭区间 [start_idx, end_idx] 的综合频率（正码+特码）。"""
    c: Counter[int] = Counter()
    for i in range(start_idx, end_idx + 1):
        for n in draws[i]["zms"]:
            c[n] += 1
        c[draws[i]["tm"]] += 1
    # 完整键覆盖
    for n in NUMBERS_RANGE:
        _ = c[n]
    return c


def classify_tm_within_window(draws: List[Dict[str, Any]], window: int) -> List[Optional[str]]:
    """基于滚动窗口，给每期 特码 分类：HOT/COLD/NEUTRAL/None。
    返回与 draws 同长度的列表。
    """
    classes: List[Optional[str]] = [None] * len(draws)
    for i in range(len(draws)):
        if i < window:
            continue
        freq = windowed_frequency(draws, i - window, i - 1)
        hot, cold = compute_hot_cold_sets(freq, HOT_COUNT, COLD_COUNT)
        tm = draws[i]["tm"]
        if tm in hot:
            classes[i] = "HOT"
        elif tm in cold:
            classes[i] = "COLD"
        else:
            classes[i] = "NEUTRAL"
    return classes


def compute_alternation_rate(classes: List[Optional[str]]) -> Tuple[int, int, float]:
    """统计 HOT/COLD 间的交替率（忽略 NEUTRAL 和 None）。"""
    prev: Optional[str] = None
    opportunities = 0
    alternations = 0
    for c in classes:
        if c not in ("HOT", "COLD"):
            continue
        if prev in ("HOT", "COLD"):
            opportunities += 1
            if c != prev:
                alternations += 1
        prev = c
    rate = (alternations / opportunities) if opportunities > 0 else 0.0
    return alternations, opportunities, rate


def last_seen_index(seq: List[int], value: int) -> Optional[int]:
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] == value:
            return i
    return None


def predict_tm_286(draws: List[Dict[str, Any]]) -> Dict[str, Any]:
    """根据滑窗频率与交替特性给出 286 期 特码预测（针对给定序列的下一期）。"""
    if not draws:
        raise ValueError("无数据，无法预测。")

    # 使用 ALT_WINDOW 交替分类
    classes = classify_tm_within_window(draws, ALT_WINDOW)
    # 统计整体交替率
    alt_count, alt_ops, alt_rate = compute_alternation_rate(classes)

    # 最近一期（285期）的分类（用窗口比它小 1 的数据）
    last_idx = len(draws) - 1
    last_class = classes[last_idx]

    # 预测 286 期期望分类
    if last_class in ("HOT", "COLD"):
        if alt_rate >= 0.5:
            expected_class = "COLD" if last_class == "HOT" else "HOT"
        else:
            expected_class = last_class
    else:
        # 若为 NEUTRAL/None，则偏向 HOT
        expected_class = "HOT"

    # 预测候选集：使用最近 RECENT_WINDOW_FOR_FREQ 的特码频率
    start_idx = max(0, len(draws) - RECENT_WINDOW_FOR_FREQ)
    tm_recent = [d["tm"] for d in draws[start_idx:]]
    tm_freq_recent = Counter(tm_recent)
    for n in NUMBERS_RANGE:
        _ = tm_freq_recent[n]

    # 根据 expected_class 提取候选
    if expected_class == "HOT":
        candidates = [n for n, _ in top_n(tm_freq_recent, HOT_COUNT)]
    else:  # COLD
        # 冷门按出现次数升序；同频次按“距离上次作为特码出现的时距”排序（越久未出越靠前）
        recent_seq_full = [d["tm"] for d in draws]  # 全量用于间隔
        def cold_key(n: int) -> Tuple[int, int]:
            freq = tm_freq_recent[n]
            last_idx_seen = last_seen_index(recent_seq_full, n)
            gap = (len(recent_seq_full) - 1 - last_idx_seen) if last_idx_seen is not None else len(recent_seq_full)
            return (freq, -gap)
        # 初步选出底部 COLD_COUNT
        sorted_by_cold = sorted(NUMBERS_RANGE, key=cold_key)
        candidates = sorted_by_cold[:COLD_COUNT]

    # 避免刚出的号码（285期的特码）
    last_tm = draws[-1]["tm"]
    candidates = [n for n in candidates if n != last_tm] or candidates

    # 打分：综合（标准化）最近频率与“未出间隔”（HOT 倾向高频，COLD 倾向大间隔）
    recent_seq_full = [d["tm"] for d in draws]
    freq_values = [tm_freq_recent[n] for n in candidates]
    # 间隔，None 视为最大
    gaps = []
    for n in candidates:
        li = last_seen_index(recent_seq_full, n)
        gap = (len(recent_seq_full) - 1 - li) if li is not None else len(recent_seq_full)
        gaps.append(gap)

    def zscore(values: List[float]) -> List[float]:
        if len(values) <= 1:
            return [0.0 for _ in values]
        mu = statistics.mean(values)
        sd = statistics.pstdev(values) or 1.0
        return [(v - mu) / sd for v in values]

    freq_z = zscore(freq_values)
    gap_z = zscore(gaps)

    scores: List[float] = []
    for i in range(len(candidates)):
        if expected_class == "HOT":
            # 高频权重大，间隔权重小（稍微正向以避免刚出）
            s = 1.0 * freq_z[i] + 0.2 * gap_z[i]
        else:  # COLD
            # 大间隔权重大，低频权重也考虑
            s = 1.0 * gap_z[i] - 0.2 * freq_z[i]
        scores.append(s)

    ranked = sorted(zip(candidates, scores, freq_values, gaps), key=lambda x: -x[1])

    prediction = ranked[0][0] if ranked else candidates[0]

    detail = {
        "last_issue": draws[-1]["issue"],
        "last_tm": draws[-1]["tm"],
        "last_class": last_class,
        "expected_class": expected_class,
        "alternations": alt_count,
        "opportunities": alt_ops,
        "alt_rate": alt_rate,
        "candidates_ranked": [
            {
                "number": n,
                "score": round(s, 4),
                "recent_freq": int(f),
                "gap_since_last_tm": int(g),
            }
            for n, s, f, g in ranked[:10]
        ],
    }

    return {"prediction": prediction, "detail": detail}


def predict_tm_ranking(draws: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成下一期特码候选排序（包含分数与交替详情）。"""
    if not draws:
        raise ValueError("无数据，无法预测。")

    classes = classify_tm_within_window(draws, ALT_WINDOW)
    alt_count, alt_ops, alt_rate = compute_alternation_rate(classes)
    last_idx = len(draws) - 1
    last_class = classes[last_idx]

    if last_class in ("HOT", "COLD"):
        expected_class = "COLD" if (alt_rate >= 0.5 and last_class == "HOT") else (
            "HOT" if (alt_rate >= 0.5 and last_class == "COLD") else last_class
        )
    else:
        expected_class = "HOT"

    start_idx = max(0, len(draws) - RECENT_WINDOW_FOR_FREQ)
    tm_recent = [d["tm"] for d in draws[start_idx:]]
    tm_freq_recent = Counter(tm_recent)
    for n in NUMBERS_RANGE:
        _ = tm_freq_recent[n]

    if expected_class == "HOT":
        candidates = [n for n, _ in top_n(tm_freq_recent, HOT_COUNT)]
    else:
        recent_seq_full = [d["tm"] for d in draws]
        def cold_key(n: int) -> Tuple[int, int]:
            freq = tm_freq_recent[n]
            last_idx_seen = last_seen_index(recent_seq_full, n)
            gap = (len(recent_seq_full) - 1 - last_idx_seen) if last_idx_seen is not None else len(recent_seq_full)
            return (freq, -gap)
        sorted_by_cold = sorted(NUMBERS_RANGE, key=cold_key)
        candidates = sorted_by_cold[:COLD_COUNT]

    last_tm = draws[-1]["tm"]
    candidates = [n for n in candidates if n != last_tm] or candidates

    recent_seq_full = [d["tm"] for d in draws]
    freq_values = [tm_freq_recent[n] for n in candidates]
    gaps = []
    for n in candidates:
        li = last_seen_index(recent_seq_full, n)
        gap = (len(recent_seq_full) - 1 - li) if li is not None else len(recent_seq_full)
        gaps.append(gap)

    def zscore(values: List[float]) -> List[float]:
        if len(values) <= 1:
            return [0.0 for _ in values]
        mu = statistics.mean(values)
        sd = statistics.pstdev(values) or 1.0
        return [(v - mu) / sd for v in values]

    freq_z = zscore(freq_values)
    gap_z = zscore(gaps)
    scores: List[float] = []
    for i in range(len(candidates)):
        if expected_class == "HOT":
            s = 1.0 * freq_z[i] + 0.2 * gap_z[i]
        else:
            s = 1.0 * gap_z[i] - 0.2 * freq_z[i]
        scores.append(s)

    ranked = sorted(zip(candidates, scores, freq_values, gaps), key=lambda x: -x[1])

    return {
        "ranked": [
            {
                "number": n,
                "score": round(s, 4),
                "recent_freq": int(f),
                "gap_since_last_tm": int(g),
            }
            for n, s, f, g in ranked
        ],
        "last_issue": draws[-1]["issue"],
        "last_tm": draws[-1]["tm"],
        "last_class": last_class,
        "expected_class": expected_class,
        "alternations": alt_count,
        "opportunities": alt_ops,
        "alt_rate": alt_rate,
    }


def render_report(draws: List[Dict[str, Any]], freqs: Dict[str, Counter], prediction: Dict[str, Any]) -> str:
    issues = [d["issue"] for d in draws]
    dates = [d.get("date") for d in draws if d.get("date")]
    date_range = (dates[0], dates[-1]) if dates else (None, None)

    def table_lines(title: str, items: List[Tuple[int, int]]) -> str:
        header = f"- **{title}**\n"
        rows = "\n".join([f"  - {num}: {cnt} 次" for num, cnt in items])
        return header + rows + "\n"

    hot_zm = top_n(freqs["zm"], HOT_COUNT)
    cold_zm = bottom_n(freqs["zm"], COLD_COUNT)
    hot_tm = top_n(freqs["tm"], HOT_COUNT)
    cold_tm = bottom_n(freqs["tm"], COLD_COUNT)
    hot_all = top_n(freqs["all"], HOT_COUNT)
    cold_all = bottom_n(freqs["all"], COLD_COUNT)

    pred_num = prediction["prediction"]
    detail = prediction["detail"]

    lines: List[str] = []
    lines.append("### 数据概览")
    lines.append(f"- **样本期数**: {len(draws)} (期号 {issues[0]} 至 {issues[-1]})")
    if date_range[0] and date_range[1]:
        lines.append(f"- **日期范围**: {date_range[0]} 至 {date_range[1]}")

    lines.append("\n### 频率统计（前10/后10）")
    lines.append(table_lines("正码 热门 (Top 10)", hot_zm))
    lines.append(table_lines("正码 冷门 (Bottom 10)", cold_zm))
    lines.append(table_lines("特码 热门 (Top 10)", hot_tm))
    lines.append(table_lines("特码 冷门 (Bottom 10)", cold_tm))
    lines.append(table_lines("综合(正+特) 热门 (Top 10)", hot_all))
    lines.append(table_lines("综合(正+特) 冷门 (Bottom 10)", cold_all))

    lines.append("\n### 交替分析（基于滑窗统计 HOT/COLD，窗口=50）")
    lines.append(f"- **上一期(第{detail['last_issue']}期) 特码**: {detail['last_tm']}，分类: {detail['last_class']}")
    lines.append(f"- **整体 HOT/COLD 交替率**: {detail['alternations']}/{detail['opportunities']} = {detail['alt_rate']:.2%}")
    lines.append(f"- **预测 286 期倾向分类**: {detail['expected_class']}")

    lines.append("\n### 286期 特码预测")
    lines.append(f"- **预测号码**: {pred_num}")
    lines.append("- **候选排序(前10)**:")
    for i, c in enumerate(detail["candidates_ranked"], start=1):
        lines.append(
            f"  {i}. {c['number']} (score={c['score']}, 最近{RECENT_WINDOW_FOR_FREQ}期频次={c['recent_freq']}, 未出间隔={c['gap_since_last_tm']})"
        )

    return "\n".join(lines) + "\n"


def analyze_last_500(draws_full: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Counter]]:
    if len(draws_full) > 500:
        subset = draws_full[-500:]
    else:
        subset = draws_full[:]
    freqs = compute_frequencies(subset)
    return subset, freqs


def tm_top_k(freqs_tm: Counter, k: int = 12) -> List[Tuple[int, int]]:
    return sorted(freqs_tm.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def main() -> None:
    # Primary page
    draws_index = parse_draws_from_html(DATA_HTML_PATH)
    # Merge with extra year pages if present
    draws = parse_draws_from_multiple([DATA_HTML_PATH] + DATA_HTML_EXTRA)
    if not draws:
        raise RuntimeError("解析失败：未提取到任何期次！请检查 data/kj_index.html 页面结构是否变化。")

    # 仅保留到 285 期（按题意），若页面多于 285 期则截取
    draws = [d for d in draws if d["issue"] <= 285]

    os.makedirs(os.path.dirname(PARSED_CSV_PATH), exist_ok=True)
    save_draws_to_csv(draws, PARSED_CSV_PATH)

    freqs = compute_frequencies(draws)
    pred = predict_tm_286(draws)

    report = render_report(draws, freqs, pred)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    # Extended analysis over last 500 draws (if available)
    draws500, freqs500 = analyze_last_500(draws)
    # Save dedicated report
    hot12_tm_500 = tm_top_k(freqs500["tm"], 12)
    hot12_all_500 = top_n(freqs500["all"], 12)
    cold12_all_500 = bottom_n(freqs500["all"], 12)
    report500_lines = [
        "### 最近500期 概览",
        f"- **样本期数**: {len(draws500)} (期号 {draws500[0]['issue']} 至 {draws500[-1]['issue']})",
        "\n### 最近500期 号码频次",
        "- **特码 Top12**",
    ]
    report500_lines += [f"  - {n}: {c} 次" for n, c in hot12_tm_500]
    report500_lines += ["\n- **综合(正+特) Top12**"]
    report500_lines += [f"  - {n}: {c} 次" for n, c in hot12_all_500]
    report500_lines += ["\n- **综合(正+特) 冷门 Bottom12**"]
    report500_lines += [f"  - {n}: {c} 次" for n, c in cold12_all_500]
    with open(REPORT_500_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report500_lines) + "\n")

    # 控制台简报
    print(json.dumps({
        "parsed_issues": [draws[0]["issue"], draws[-1]["issue"]],
        "num_draws": len(draws),
        "prediction_tm_286": pred["prediction"],
        "alt_rate": pred["detail"]["alt_rate"],
        "expected_class": pred["detail"]["expected_class"],
        "report_path": REPORT_PATH,
        "report_500_path": REPORT_500_PATH,
        "parsed_csv": PARSED_CSV_PATH,
        "tm_top12_last500": tm_top_k(freqs500["tm"], 12),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
