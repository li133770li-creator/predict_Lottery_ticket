# -*- coding: utf-8 -*-
import os, re, math, random, sys
from collections import Counter, defaultdict

RAW_DIR = os.path.join("data","amlhc","raw")
KJ_PATH = os.path.join(RAW_DIR, "kj_2025.html")
SX_PATH = os.path.join(RAW_DIR, "kj_sx.html")
TMSX_PATH = os.path.join(RAW_DIR, "kj_tmsxzs_2025.html")

if not os.path.exists(KJ_PATH) or not os.path.exists(SX_PATH):
    print("Missing required HTML files. Ensure data/amlhc/raw/kj_2025.html and kj_sx.html exist.")
    sys.exit(1)

with open(KJ_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    kj_html = f.read()
with open(SX_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    sx_html = f.read()

# Parse draws for 2025: 292期, each block has 7 numbers inside <dt class="ball-...">NN</dt>
# preceded by title with issue number.

titles = list(re.finditer(r"<div class=\"kj-tit\">[^<]*?第<span class=\"text-blue text-strong\">(\d+)</span>期</div>", kj_html))

draws = []  # list of (issue:int, nums:[int]*7)
for m in titles:
    issue = int(m.group(1))
    start = m.end()
    box_m = re.search(r"<div class=\"kj-box\">(.*?)</div>", kj_html[start:], flags=re.S)
    if not box_m:
        continue
    box_html = box_m.group(1)
    items = re.findall(r"<dt class=\"ball-(?:red|green|blue)\">\s*(\d{1,2})\s*</dt>", box_html)
    if len(items) >= 7:
        nums = list(map(int, items[:7]))
        draws.append((issue, nums))

# Sort ascending by issue
if not draws:
    print("No draws parsed.")
    sys.exit(1)

draws.sort(key=lambda x: x[0])
issues = [i for i,_ in draws]

# Parse attributes (zodiac and colors) from sx page
# Zodiac mapping: under sections with <span class="sx">生肖...</span> ... numbers until next section

zodiac_order = ["鼠","牛","虎","兔","龙","蛇","马","羊","猴","鸡","狗","猪"]
num_to_zodiac = {}
for zx in zodiac_order:
    # find section following the span sx for current zodiac
    sec_iter = re.finditer(rf"<span class=\"sx\">{zx}[^<]*</span>([\s\S]*?)(?:<span class=\"sx\">|</li>)", sx_html)
    for sm in sec_iter:
        sec = sm.group(1)
        for n in re.findall(r"(\d{1,2})", sec):
            num_to_zodiac[int(n)] = zx

num_to_color = {}
for cname, key in [("红波","red"),("蓝波","blue"),("绿波","green")]:
    cm = re.search(rf">{cname}<.*?</td>\s*<td>([\s\S]*?)</td>", sx_html)
    if cm:
        for n in re.findall(r"(\d{1,2})", cm.group(1)):
            num_to_color[int(n)] = key

# Utility calculations

def last_n_draws(n):
    if n >= len(draws):
        return draws[:]
    return draws[-n:]

def freq_counter(window=None):
    sub = draws if window is None else last_n_draws(window)
    c = Counter()
    for _, nums in sub:
        c.update(nums)
    return c

def omission_all():
    last_seen = {n:-1 for n in range(1,50)}
    for issue, nums in draws:
        for v in set(nums):
            last_seen[v] = issue
    max_issue = draws[-1][0]
    return {n:(max_issue - last_seen[n] if last_seen[n]!=-1 else max_issue) for n in range(1,50)}

def ema_scores(span=20):
    alpha = 2.0/(span+1.0)
    scores = {n:0.0 for n in range(1,50)}
    for _, nums in draws:
        s = set(nums)
        for n in range(1,50):
            x = 1.0 if n in s else 0.0
            scores[n] = alpha * x + (1.0-alpha) * scores[n]
    return scores

def cooccurrence():
    # return dict of dict for sparse co-occur counts
    mat = {n:{ } for n in range(1,50)}
    for _, nums in draws:
        s = set(nums)
        for a in s:
            for b in s:
                if a!=b:
                    mat[a][b] = mat[a].get(b,0) + 1
    return mat

# Balanced picking by (color, odd/even)

def pick_balanced(candidates, k=7):
    buckets = defaultdict(list)
    for n in candidates:
        color = num_to_color.get(n, '')
        parity = '单' if (n%2==1) else '双'
        buckets[(color,parity)].append(n)
    # Ensure deterministic order within buckets
    for b in buckets:
        buckets[b].sort()
    picks = []
    keys = sorted(buckets.keys(), key=lambda b: -len(buckets[b]))
    while len(picks)<k and any(buckets.values()):
        progressed=False
        for bk in keys:
            if buckets[bk]:
                x=buckets[bk].pop(0)
                if x not in picks:
                    picks.append(x)
                    progressed=True
                if len(picks)>=k:
                    break
        if not progressed:
            break
    for n in candidates:
        if len(picks)>=k:
            break
        if n not in picks:
            picks.append(n)
    return picks[:k]

# Strategies

freq120 = freq_counter(120)
freq_candidates = [n for n,_ in freq120.most_common(20)]
freq_balanced = pick_balanced(freq_candidates, 7)

om = omission_all()
omission_sorted = sorted(om.items(), key=lambda x:(-x[1], x[0]))
omission_top = [n for n,_ in omission_sorted[:20]]
omission_picks = pick_balanced(omission_top, 7)

ema = ema_scores(span=20)
trend_sorted = sorted(ema.items(), key=lambda x:(-x[1], x[0]))
trend_top = [n for n,_ in trend_sorted[:20]]
trend_picks = pick_balanced(trend_top, 7)

co = cooccurrence()
last_nums = set(draws[-1][1])
assoc_scores = {}
for n in range(1,50):
    if n in last_nums:
        continue
    assoc_scores[n] = sum(co.get(n,{}).get(m,0) for m in last_nums)
assoc_sorted = sorted(assoc_scores.items(), key=lambda x:(-x[1], x[0]))
assoc_top = [n for n,_ in assoc_sorted[:20]]
assoc_picks = pick_balanced(assoc_top, 7)

# Clustering via simple k-means using cosine similarity on co-occurrence rows (pure python)

def row_vector(n):
    row = co.get(n, {})
    # Build dense list 1..49 (ignore self)
    vec = [0.0]*49
    for m,v in row.items():
        if 1<=m<=49:
            vec[m-1] = float(v)
    # zero self
    vec[n-1] = 0.0
    # L2 normalize
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x/norm for x in vec]

nums = list(range(1,50))
X = {n: row_vector(n) for n in nums}
random.seed(42)
centers = [X[n] for n in random.sample(nums, 6)]
labels = {n:0 for n in nums}
for _ in range(15):
    # assign
    for n in nums:
        sims = [sum(X[n][i]*c[i] for i in range(49)) for c in centers]
        labels[n] = max(range(6), key=lambda j: sims[j])
    # update
    new_centers = []
    for j in range(6):
        members = [n for n in nums if labels[n]==j]
        if not members:
            new_centers.append(centers[j])
            continue
        acc = [0.0]*49
        for n in members:
            v = X[n]
            for i in range(49):
                acc[i]+=v[i]
        # normalize
        norm = math.sqrt(sum(x*x for x in acc)) or 1.0
        new_centers.append([x/norm for x in acc])
    centers = new_centers

clusters = defaultdict(list)
for n in nums:
    clusters[labels[n]].append(n)

full_freq = freq_counter(None)
cluster_picks = []
for cid, members in clusters.items():
    members.sort()
    members_by_freq = sorted(members, key=lambda n:(-full_freq.get(n,0), n))
    if members_by_freq:
        cluster_picks.append(members_by_freq[0])
cluster_picks = sorted(cluster_picks, key=lambda n:(-full_freq.get(n,0), n))[:7]

# Simple ML ranking (blend recent freq and omission, z-scored)

def zscore(values):
    mean = sum(values)/len(values)
    var = sum((x-mean)**2 for x in values)/len(values)
    std = math.sqrt(var) or 1.0
    return [(x-mean)/std for x in values]

freq50 = freq_counter(50)
base = [freq50.get(n,0) for n in nums]
omvals = [om.get(n,0) for n in nums]
zb = zscore(base)
zo = zscore(omvals)
ml_scores = {n: 0.6*zb[i] + 0.4*zo[i] for i,n in enumerate(nums)}
ml_sorted = sorted(ml_scores.items(), key=lambda x:(-x[1], x[0]))
ml_picks = [n for n,_ in ml_sorted[:7]]

# Choose TM: highest EMA among picks (fallback to highest omission if ties)

def pick_tm(picks):
    best = None
    for n in picks:
        score = (ema.get(n,0.0), om.get(n,0))
        if best is None or score>best[0]:
            best = (score, n)
    return best[1] if best else picks[0]

def zodiac_from_results(n):
    # Try zero-padded match first
    m = re.search(rf"<dt class=\"ball-(?:red|green|blue)\">\s*{n:02d}\s*</dt>\s*<dd>([^<])", kj_html)
    if not m:
        m = re.search(rf"<dt class=\"ball-(?:red|green|blue)\">\s*{n}\s*</dt>\s*<dd>([^<])", kj_html)
    if m:
        return m.group(1)
    # Fallback to zodiac table mapping if available
    return num_to_zodiac.get(n, '')

strategies = [
    ("频率平衡预测", freq_balanced),
    ("遗漏值预测", omission_picks),
    ("趋势预测", trend_picks),
    ("关联性预测", assoc_picks),
    ("聚类预测", cluster_picks),
    ("机器学习预测", ml_picks),
]

for name, picks in strategies:
    tm = pick_tm(picks)
    zx = zodiac_from_results(tm)
    print(f"{name} 特码: {tm} ({zx})  推荐: {picks}")

# If TM zodiac trend page exists, compute trend-adjusted TOP5 for each strategy
def normalize(values_dict):
    vals = list(values_dict.values())
    if not vals:
        return {k:0.0 for k in values_dict}
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-9:
        return {k:0.5 for k in values_dict}
    return {k:(v - vmin)/(vmax - vmin) for k, v in values_dict.items()}

zx_recent = {z:0 for z in ["鼠","牛","虎","兔","龙","蛇","马","羊","猴","鸡","狗","猪"]}
if os.path.exists(TMSX_PATH):
    try:
        with open(TMSX_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            tmsx_html = f.read()
        # Extract rows and keep last 50 issues' zodiac (based on 'tema' cell position)
        rows = []
        for m in re.finditer(r"<tr>\s*<td>(\d{1,3})</td>([\s\S]*?)</tr>", tmsx_html):
            issue = int(m.group(1))
            cells = m.group(2)
            tds = re.findall(r"<td[\s\S]*?>[\s\S]*?</td>", cells)
            idx = None
            for i, td in enumerate(tds):
                if 'tema' in td:
                    idx = i
                    break
            if idx is None:
                continue
            zodiac_order = ["鼠","牛","虎","兔","龙","蛇","马","羊","猴","鸡","狗","猪"]
            if 0 <= idx < len(zodiac_order):
                rows.append((issue, zodiac_order[idx]))
        rows.sort(key=lambda x: x[0])
        last50 = [zx for _, zx in rows[-50:]]
        for z in last50:
            zx_recent[z] = zx_recent.get(z, 0) + 1
    except Exception:
        pass

ema_norm = normalize({n: ema.get(n,0.0) for n in range(1,50)})
om_norm = normalize({n: om.get(n,0) for n in range(1,50)})
zx_norm = {}
for n in range(1,50):
    z = num_to_zodiac.get(n, '')
    zx_norm[n] = zx_recent.get(z, 0)
zx_norm = normalize(zx_norm)

def score_number(n):
    return 0.6*ema_norm.get(n,0.0) + 0.3*zx_norm.get(n,0.0) + 0.1*om_norm.get(n,0.0)

print("TOP5 (趋势加权)")
for name, picks in strategies:
    ranked = sorted(picks, key=lambda n: (-score_number(n), n))[:5]
    labels = [f"{n}({zodiac_from_results(n)})" for n in ranked]
    print(f"{name} TOP5: {labels}")
