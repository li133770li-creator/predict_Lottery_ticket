# -*- coding: utf-8 -*-
import re
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict

# input files newest->older
FILES = [
    '/workspace/data/kj_2025.html',
    '/workspace/data/kj_2024.html',
    '/workspace/data/kj_2023.html',
]


def parse_page(path: str) -> List[Dict]:
    html = Path(path).read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'html.parser')
    records = []
    for title_div in soup.select('.kj-tit'):
        issue_text = title_div.get_text(' ', strip=True)
        m = re.search(r'第\s*(\d{1,3})\s*期', issue_text)
        issue = int(m.group(1)) if m else None
        box_div = title_div.find_next_sibling('div', class_='kj-box')
        if not box_div:
            continue
        # numbers include 6 正码 + 1 特码; we want all 7 for frequency
        nums = []
        for li in box_div.select('ul > li'):
            if 'kj-jia' in (li.get('class') or []):
                continue
            dt = li.select_one('dt')
            if not dt:
                continue
            num_txt = dt.get_text(strip=True)
            if not num_txt:
                continue
            try:
                nums.append(int(num_txt))
            except ValueError:
                continue
        if len(nums) == 7:
            records.append({'issue': issue, 'nums': nums})
    # Page lists newest first; keep that order
    return records


def collect_last_n(n: int) -> List[Dict]:
    all_records = []
    for f in FILES:
        p = Path(f)
        if not p.exists():
            continue
        recs = parse_page(str(p))
        all_records.extend(recs)
    # Ensure overall newest->older ordering already
    # Take first n
    return all_records[:n]


def main():
    last500 = collect_last_n(500)
    print(f'collected_draws {len(last500)}')
    # frequency 1..49
    from collections import Counter
    freq = Counter()
    for rec in last500:
        for x in rec['nums']:
            if 1 <= x <= 49:
                freq[x] += 1
    # print sorted lists
    hot = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:10]
    cold = sorted(freq.items(), key=lambda x: (x[1], x[0]))[:10]
    print('hot', hot)
    print('cold', cold)

if __name__ == '__main__':
    main()
