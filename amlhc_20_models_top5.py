# -*- coding: utf-8 -*-
import os, re, math, random, sys
from collections import Counter, defaultdict

RAW_DIR = os.path.join('data','amlhc','raw')
KJ_2025 = os.path.join(RAW_DIR, 'kj_2025.html')
SX_2025 = os.path.join(RAW_DIR, 'kj_sx.html')
TMSX_2025 = os.path.join(RAW_DIR, 'kj_tmsxzs_2025.html')

for p in [KJ_2025, SX_2025, TMSX_2025]:
    if not os.path.exists(p):
        print('Missing', p)
        sys.exit(1)

kj_html = open(KJ_2025, 'r', encoding='utf-8', errors='ignore').read()
sx_html = open(SX_2025, 'r', encoding='utf-8', errors='ignore').read()
tmsx_html = open(TMSX_2025, 'r', encoding='utf-8', errors='ignore').read()

# Parse draws, extract TM series (7th number)

titles = list(re.finditer(r"<div class=\"kj-tit\">[^<]*?第<span class=\"text-blue text-strong\">(\d+)</span>期</div>", kj_html))
draws = []
for m in titles:
    issue = int(m.group(1))
    start = m.end()
    box_m = re.search(r"<div class=\"kj-box\">(.*?)</div>", kj_html[start:], flags=re.S)
    if not box_m:
        continue
    items = re.findall(r"<dt class=\"ball-(?:red|green|blue)\">\s*(\d{1,2})\s*</dt>", box_m.group(1))
    if len(items) >= 7:
        nums = list(map(int, items[:7]))
        draws.append((issue, nums))

if not draws:
    print('No draws parsed from 2025 page.')
    sys.exit(1)

# Sort and build series

draws.sort(key=lambda x: x[0])
issues = [i for i,_ in draws]
tm_series = [nums[6] for _, nums in draws]
N = len(tm_series)

# Mapping: number -> zodiac/color/five-elements from 2025 attributes page

zodiac_order = ["鼠","牛","虎","兔","龙","蛇","马","羊","猴","鸡","狗","猪"]
num_to_zodiac = {}
for zx in zodiac_order:
    pattern = r"<span class=\"sx\">"+zx+r"[^<]*</span>([\s\S]*?)(?:<span class=\"sx\"|</li>|</ul>)"
    for sm in re.finditer(pattern, sx_html):
        sec = sm.group(1)
        for n in re.findall(r"(\d{1,2})", sec):
            num_to_zodiac[int(n)] = zx

num_to_color = {}
for cname, key in [("红波","red"),("蓝波","blue"),("绿波","green")]:
    cm = re.search(r">"+cname+"<.*?</td>\s*<td>([\s\S]*?)</td>", sx_html)
    if cm:
        for n in re.findall(r"(\d{1,2})", cm.group(1)):
            num_to_color[int(n)] = key

num_to_element = {}
for element in ["金","木","水","火","土"]:
    sec = re.search(r">"+element+"<.*?</td>\s*([\s\S]*?)</tr>", sx_html)
    if sec:
        for n in re.findall(r"(\d{1,2})", sec.group(1)):
            num_to_element[int(n)] = element

# Helpers

def last_n(seq, n):
    return seq[-n:] if n < len(seq) else list(seq)

def freq_counter(seq):
    return Counter(seq)

def omission(seq):
    last_seen = {n:-1 for n in range(1,50)}
    for i, x in enumerate(seq):
        last_seen[x] = i
    return {n: (len(seq)-1 - last_seen[n] if last_seen[n] != -1 else len(seq)) for n in range(1,50)}

def ema_scores(seq, span=20):
    alpha = 2.0/(span+1.0)
    scores = {n:0.0 for n in range(1,50)}
    for x in seq:
        present = x
        for n in range(1,50):
            val = 1.0 if n == present else 0.0
            scores[n] = alpha*val + (1.0-alpha)*scores[n]
    return scores

def zscore_dict(d):
    vals = list(d.values())
    if not vals:
        return {k:0.0 for k in d}
    mean = sum(vals)/len(vals)
    var = sum((v-mean)**2 for v in vals)/len(vals)
    std = math.sqrt(var) or 1.0
    return {k:(v-mean)/std for k,v in d.items()}

def minmax_norm(d):
    if not d:
        return d
    vmin = min(d.values())
    vmax = max(d.values())
    if abs(vmax - vmin) < 1e-12:
        return {k:0.5 for k in d}
    return {k:(v-vmin)/(vmax-vmin) for k,v in d.items()}

# Transition matrix for TM numbers (1-step Markov)

trans_counts = {i: Counter() for i in range(1,50)}
for a, b in zip(tm_series[:-1], tm_series[1:]):
    trans_counts[a][b] += 1

# Zodiac series from tmsx page

rows = []
for m in re.finditer(r"<tr>\s*<td>(\d{1,3})</td>([\s\S]*?)</tr>", tmsx_html):
    issue = int(m.group(1))
    cells = m.group(2)
    tds = re.findall(r"<td[\s\S]*?>[\s\S]*?</td>", cells)
    idx = None
    for i, td in enumerate(tds):
        if 'tema' in td:
            idx = i; break
    if idx is None:
        continue
    if 0 <= idx < 12:
        rows.append((issue, zodiac_order[idx]))
rows.sort(key=lambda x:x[0])
zx_series = [z for _, z in rows]
# Zodiac transition counts
zx_trans = defaultdict(lambda: Counter())
for a, b in zip(zx_series[:-1], zx_series[1:]):
    zx_trans[a][b] += 1
last_zx = zx_series[-1] if zx_series else ''

# Base stats

freq_full = freq_counter(tm_series)
freq120 = freq_counter(last_n(tm_series, 120))
freq80 = freq_counter(last_n(tm_series, 80))
freq50 = freq_counter(last_n(tm_series, 50))
freq15 = freq_counter(last_n(tm_series, 15))

om = omission(tm_series)
ema = ema_scores(tm_series, span=20)

def topk_from_scores(scores, k=5):
    return [n for n,_ in sorted(scores.items(), key=lambda x:(-x[1], x[0]))[:k]]

# Models

models = {}

# 1 频率平衡预测: round-robin by color+parity from freq120
buckets = defaultdict(list)
for n in range(1,50):
    buckets[(num_to_color.get(n,''), '单' if n%2==1 else '双')].append(n)
# sort each bucket by freq120 desc
for bk in buckets:
    buckets[bk] = sorted(buckets[bk], key=lambda n:(-freq120.get(n,0), n))
res=[]
keys = sorted(buckets.keys(), key=lambda b: -sum(freq120.get(n,0) for n in buckets[b]))
while len(res)<5 and any(buckets.values()):
    for bk in keys:
        if buckets[bk]:
            x = buckets[bk].pop(0)
            if x not in res:
                res.append(x)
            if len(res)>=5: break
models['频率平衡预测'] = res

# 2 遗漏值预测: largest omission
models['遗漏值预测'] = [n for n,_ in sorted(om.items(), key=lambda x:(-x[1], x[0]))[:5]]

# 3 趋势预测: highest EMA
models['趋势预测'] = topk_from_scores(ema, 5)

# 4 关联性预测: transition from last TM
last_tm = tm_series[-1]
row = trans_counts.get(last_tm, {})
# Laplace smoothing with small epsilon
cands = {n: row.get(n,0) + 1e-6 for n in range(1,50) if n!=last_tm}
models['关联性预测'] = [n for n,_ in sorted(cands.items(), key=lambda x:(-x[1], x[0]))[:5]]

# 5 聚类预测: k-means on transition rows, pick top cluster by freq120
random.seed(42)
X = {n: [float(trans_counts[n].get(m,0)) for m in range(1,50)] for n in range(1,50)}
# L2 norm
for n in X:
    vec = X[n]
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    X[n] = [v/norm for v in vec]
K=5
centers=[X[n] for n in random.sample(range(1,50), K)]
labels={n:0 for n in range(1,50)}
for _ in range(10):
    # assign
    for n in range(1,50):
        sims=[sum(X[n][i]*c[i] for i in range(49)) for c in centers]
        labels[n]=max(range(K), key=lambda j: sims[j])
    # update
    new=[]
    for j in range(K):
        members=[n for n in range(1,50) if labels[n]==j]
        if not members:
            new.append(centers[j])
        else:
            acc=[0.0]*49
            for n in members:
                for i,v in enumerate(X[n]): acc[i]+=v
            norm=math.sqrt(sum(v*v for v in acc)) or 1.0
            new.append([v/norm for v in acc])
    centers=new
# choose cluster with highest avg freq120
cluster_scores=defaultdict(float)
for j in range(K):
    members=[n for n in range(1,50) if labels[n]==j]
    if members:
        cluster_scores[j]=sum(freq120.get(n,0) for n in members)/len(members)
best_cluster=max(cluster_scores, key=lambda j: cluster_scores[j]) if cluster_scores else 0
cand=[n for n in range(1,50) if labels[n]==best_cluster]
models['聚类预测'] = [n for n in sorted(cand, key=lambda n:(-freq120.get(n,0), n))[:5]]

# 6 机器学习预测: zscore(freq50)*0.6 + zscore(om)*0.4
z_freq50 = zscore_dict({n: freq50.get(n,0) for n in range(1,50)})
z_om = zscore_dict(om)
scores_ml = {n: 0.6*z_freq50.get(n,0.0) + 0.4*z_om.get(n,0.0) for n in range(1,50)}
models['机器学习预测'] = topk_from_scores(scores_ml, 5)

# 7 频率分析模型: weighted multi-window frequency
weights = [(freq_full, 0.4), (freq120, 0.25), (freq80, 0.2), (freq50, 0.1), (freq15, 0.05)]
scores = defaultdict(float)
for d, w in weights:
    mx = max(d.values()) if d else 1
    for n in range(1,50):
        scores[n] += w * (d.get(n,0)/(mx or 1))
models['频率分析模型'] = topk_from_scores(scores, 5)

# 8 统计学模型: z-score of full frequency vs uniform
p = 1.0/49.0
mu = N*p
sigma = math.sqrt(N*p*(1-p)) or 1.0
z_full = {n: (freq_full.get(n,0) - mu)/sigma for n in range(1,50)}
models['统计学模型'] = topk_from_scores(z_full, 5)

# 9 ARIMA模型(简化): mean of last diffs -> target center; rank by proximity + freq50
diffs=[b-a for a,b in zip(tm_series[:-1], tm_series[1:])]
mean_diff = sum(diffs[-20:])/min(20, len(diffs)) if diffs else 0
forecast = (tm_series[-1] + mean_diff)
# circular distance on 1..49
scores_arima = {}
for n in range(1,50):
    dist = min(abs(n-forecast), 49-abs(n-forecast))
    scores_arima[n] = (1.0/(1.0+dist)) + 0.2*(freq50.get(n,0))
models['ARIMA模型预测'] = topk_from_scores(scores_arima, 5)

# 10 LSTM适配(回退n-gram): order-3 -> 2 -> 1 transition
order3 = defaultdict(lambda: Counter())
for i in range(3, N):
    key=(tm_series[i-3], tm_series[i-2], tm_series[i-1])
    order3[key][tm_series[i]]+=1
key=(tm_series[-3], tm_series[-2], tm_series[-1])
row3 = order3.get(key, {})
if row3:
    cand = dict(row3)
else:
    order2 = defaultdict(lambda: Counter())
    for i in range(2, N):
        k=(tm_series[i-2], tm_series[i-1])
        order2[k][tm_series[i]]+=1
    row2 = order2.get((tm_series[-2], tm_series[-1]), {})
    cand = dict(row2) if row2 else dict(trans_counts.get(tm_series[-1], {}))
for n in range(1,50):
    cand[n] = cand.get(n, 0)
models['LSTM模型适配预测'] = [n for n,_ in sorted(cand.items(), key=lambda x:(-x[1], x[0]))[:5]]

# 11 统计检验(偏差与streak): break current parity streak + high omission
# compute parity streak on TM series
last_parity = tm_series[-1] % 2
streak=1
for x in tm_series[-2::-1]:
    if x%2 == last_parity:
        streak+=1
    else:
        break
target_parity = 1-last_parity if streak>=3 else last_parity  # flip if long streak
cand=[n for n in range(1,50) if (n%2)==target_parity]
models['统计检验模型预测'] = [n for n in sorted(cand, key=lambda n:(-om.get(n,0), -freq50.get(n,0), n))[:5]]

# 12 贝叶斯后验概率: Beta(1,1) posterior mean + recency boost
post_mean = {n: (freq_full.get(n,0)+1.0)/(N+2.0) for n in range(1,50)}
rec = {n: freq15.get(n,0)/max(1, min(15,N)) for n in range(1,50)}
scores_bayes = {n: post_mean[n] + 0.5*rec[n] for n in range(1,50)}
models['贝叶斯后验概率模型预测'] = topk_from_scores(scores_bayes, 5)

# 13 马尔科夫链两阶段: zodiac->numbers within zx by last120 freq
next_zx = zx_trans[last_zx].most_common(1)[0][0] if zx_trans[last_zx] else ''
zx_cands = [n for n in range(1,50) if num_to_zodiac.get(n,'')==next_zx]
models['马尔科夫链高级二阶段模型预测'] = [n for n,_ in sorted(zx_cands, key=lambda n:(-freq120.get(n,0), -om.get(n,0), n))[:5]] if zx_cands else []

# 14 动态概率矩阵: head-tail heat last60
last60 = last_n(tm_series, 60)
head_cnt = Counter([x//10 for x in last60])
tail_cnt = Counter([x%10 for x in last60])
head_tot = sum(head_cnt.values()) or 1
tail_tot = sum(tail_cnt.values()) or 1
scores_dyn = {}
for n in range(1,50):
    h = n//10
    t = n%10
    scores_dyn[n] = (head_cnt.get(h,0)/head_tot) * (tail_cnt.get(t,0)/tail_tot)
models['动态概率矩阵模型预测'] = topk_from_scores(scores_dyn, 5)

# 15 LSTM神经网络(核密度近似): decayed kernel around last 12 TMs
lastK = last_n(tm_series, 12)
weights = [math.exp(-0.2*(len(lastK)-1 - i)) for i in range(len(lastK))]
# circular distance kernel
scores_rnn = {n:0.0 for n in range(1,50)}
for i,x in enumerate(lastK):
    for n in range(1,50):
        d = min(abs(n-x), 49-abs(n-x))
        scores_rnn[n] += weights[i] * math.exp(-0.5*(d/3.0)**2)
models['LSTM神经网络模型预测'] = topk_from_scores(scores_rnn, 5)

# 16 逐期解析: enforce pattern from last 15 (color majority and odd-even majority)
colors = [num_to_color.get(x,'') for x in last_n(tm_series, 15)]
parities = [x%2 for x in last_n(tm_series, 15)]
color_major = Counter(colors).most_common(1)[0][0] if colors else ''
parity_major = Counter(parities).most_common(1)[0][0] if parities else 0
cand=[n for n in range(1,50) if num_to_color.get(n,'')==color_major and (n%2)==parity_major]
models['逐期解析模型预测'] = [n for n in sorted(cand, key=lambda n:(-freq50.get(n,0), -om.get(n,0), n))[:5]]

# 17 精确频率统计: full frequency top5 excluding last TM
cand = [n for n,_ in freq_full.most_common() if n != tm_series[-1]]
models['精确频率统计模型预测'] = cand[:5]

# 18 转移矩阵模型: smoothed row of last TM
row = trans_counts.get(tm_series[-1], {})
row_s = {n: row.get(n,0)+1 for n in range(1,50) if n!=tm_series[-1]}
models['转移矩阵模型预测'] = [n for n,_ in sorted(row_s.items(), key=lambda x:(-x[1], x[0]))[:5]]

# 19 Python综合模型: blend of freq analysis, EMA, Bayes
sA = scores
sE = minmax_norm(ema)
sB = minmax_norm(scores_bayes)
comb = {n: 0.5*sA.get(n,0.0) + 0.3*sE.get(n,0.0) + 0.2*sB.get(n,0.0) for n in range(1,50)}
models['Python模型预测'] = topk_from_scores(comb, 5)

# 20 生肖比例模型: track current->next zodiac ratio
next_zx = zx_trans[last_zx].most_common(1)[0][0] if zx_trans[last_zx] else ''
cand=[n for n in range(1,50) if num_to_zodiac.get(n,'')==next_zx]
models['生肖比例模型预测'] = [n for n,_ in sorted(cand, key=lambda n:(-freq50.get(n,0), -om.get(n,0), n))[:5]] if cand else []

# Output
order = [
    '频率平衡预测','遗漏值预测','趋势预测','关联性预测','聚类预测','机器学习预测',
    '频率分析模型','统计学模型','ARIMA模型预测','LSTM模型适配预测','统计检验模型预测','贝叶斯后验概率模型预测',
    '马尔科夫链高级二阶段模型预测','动态概率矩阵模型预测','LSTM神经网络模型预测','逐期解析模型预测','精确频率统计模型预测','转移矩阵模型预测','Python模型预测','生肖比例模型预测'
]
for k in order:
    vals = models.get(k, [])
    label = ', '.join(str(x) for x in vals[:5]) if vals else '[]'
    print(f"{k} TOP5: {label}")
