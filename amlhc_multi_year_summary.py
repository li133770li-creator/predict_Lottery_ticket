# -*- coding: utf-8 -*-
import os, re, sys
from collections import Counter

RAW_DIR = os.path.join('data','amlhc','raw')
files = [
    ('2023', 'kj_2023.html', 'kj_sx_2023.html'),
    ('2024', 'kj_2024.html', 'kj_sx_2024.html'),
    ('2025', 'kj_2025.html', 'kj_sx.html'),
]

num_to_zodiac_by_year = {}

def parse_zx_map(sx_html):
    zodiac_order = ["鼠","牛","虎","兔","龙","蛇","马","羊","猴","鸡","狗","猪"]
    mp = {}
    for zx in zodiac_order:
        for m in re.finditer(rf"<span class=\"sx\">{zx}[^<]*</span>([\s\S]*?)(?:<span class=\"sx\"|</li>|</ul>)", sx_html):
            sec = m.group(1)
            for n in re.findall(r"(\d{1,2})", sec):
                mp[int(n)] = zx
    return mp

rows_all = []
for year, kj_file, sx_file in files:
    kj_path = os.path.join(RAW_DIR, kj_file)
    sx_path = os.path.join(RAW_DIR, sx_file)
    if not (os.path.exists(kj_path) and os.path.exists(sx_path)):
        continue
    with open(sx_path,'r',encoding='utf-8',errors='ignore') as f:
        zx_map = parse_zx_map(f.read())
    num_to_zodiac_by_year[year]=zx_map
    with open(kj_path,'r',encoding='utf-8',errors='ignore') as f:
        html = f.read()
    titles = list(re.finditer(r"<div class=\"kj-tit\">[^<]*?第<span class=\"text-blue text-strong\">(\d+)</span>期</div>", html))
    draws = []
    for m in titles:
        issue = int(m.group(1))
        start = m.end()
        box_m = re.search(r"<div class=\"kj-box\">(.*?)</div>", html[start:], flags=re.S)
        if not box_m:
            continue
        box_html = box_m.group(1)
        items = re.findall(r"<dt class=\"ball-(?:red|green|blue)\">\s*(\d{1,2})\s*</dt>", box_html)
        if len(items)>=7:
            nums = list(map(int, items[:7]))
            draws.append((issue, nums))
    draws.sort(key=lambda x:x[0])
    for issue, nums in draws:
        tm = nums[6]
        zx = zx_map.get(tm, '')
        rows_all.append((year, issue, tm, zx))

# Aggregate 2023-2025 up to 2025-292
rows_all.sort(key=lambda r: (r[0], r[1]))
# limit 2025 to 292 issues
rows_all = [r for r in rows_all if not (r[0]=='2025' and r[1]>292)]

cnt_tm = Counter([tm for _,_,tm,_ in rows_all])
cnt_zx = Counter([zx for *_, zx in rows_all if zx])

print('TM频次TOP10:', cnt_tm.most_common(10))
print('生肖频次:', ','.join(f"{k}:{cnt_zx[k]}" for k in ["鼠","牛","虎","兔","龙","蛇","马","羊","猴","鸡","狗","猪"]))
