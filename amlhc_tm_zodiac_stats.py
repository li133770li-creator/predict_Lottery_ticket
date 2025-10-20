# -*- coding: utf-8 -*-
import os, re, sys
from collections import Counter

RAW_DIR = os.path.join("data","amlhc","raw")
TMSX_PATH = os.path.join(RAW_DIR, "kj_tmsxzs_2025.html")

if not os.path.exists(TMSX_PATH):
    print("Missing:", TMSX_PATH)
    sys.exit(1)

with open(TMSX_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    html = f.read()

# Zodiac order in header (observed on page)
zodiac_order = ["鼠","牛","虎","兔","龙","蛇","马","羊","猴","鸡","狗","猪"]

# Find the item table rows
# We'll scan rows that start with <tr><td>ISSUE</td> and then 12 tds, exactly one contains class 'tema'
rows = []
for m in re.finditer(r"<tr>\s*<td>(\d{1,3})</td>([\s\S]*?)</tr>", html):
    issue = int(m.group(1))
    cells = m.group(2)
    # Extract tds that correspond to zodiac columns (exclude the first issue td)
    tds = re.findall(r"<td[\s\S]*?>[\s\S]*?</td>", cells)
    # Sanity check: look for 'tema' cell
    idx = None
    for i, td in enumerate(tds):
        if 'tema' in td:
            idx = i
            break
    if idx is None:
        continue
    # Map i -> zodiac; i should be 0..11 aligning to zodiac_order
    if 0 <= idx < len(zodiac_order):
        rows.append((issue, zodiac_order[idx]))

# Sort by issue ascending
rows.sort(key=lambda x: x[0])

cnt_all = Counter([zx for _, zx in rows])
lastN = 50
cnt_50 = Counter([zx for _, zx in rows[-lastN:]])

# Print summary
print("ALL", ",".join(f"{k}:{cnt_all[k]}" for k in zodiac_order))
print("LAST50", ",".join(f"{k}:{cnt_50[k]}" for k in zodiac_order))
