# -*- coding: utf-8 -*-
import os, re, math, random
from collections import Counter, defaultdict

RAW_DIR = os.path.join('data','amlhc','raw')
KJ_2025 = os.path.join(RAW_DIR, 'kj_2025.html')
SX_2025 = os.path.join(RAW_DIR, 'kj_sx.html')
TMSX_2025 = os.path.join(RAW_DIR, 'kj_tmsxzs_2025.html')

# Load required pages
for p in [KJ_2025, SX_2025]:
    if not os.path.exists(p):
        raise SystemExit('Missing '+p)
kj_html = open(KJ_2025, 'r', encoding='utf-8', errors='ignore').read()
sx_html = open(SX_2025, 'r', encoding='utf-8', errors='ignore').read()
tmsx_html = open(TMSX_2025, 'r', encoding='utf-8', errors='ignore').read() if os.path.exists(TMSX_2025) else ''

# Parse draws (2025)
M=list(re.finditer(r'<div class="kj-tit">[^<]*?第<span class="text-blue text-strong">(\d+)</span>期</div>', kj_html))
draws=[]
for m in M:
    issue=int(m.group(1))
    b=re.search(r'<div class="kj-box">(.*?)</div>', kj_html[m.end():], flags=re.S)
    if not b: continue
    nums=[int(x) for x in re.findall(r'<dt class="ball-(?:red|green|blue)\">\s*(\d{1,2})\s*</dt>', b.group(1))]
    if len(nums)>=7:
        draws.append((issue, nums))

draws.sort(key=lambda x:x[0])
issues=[i for i,_ in draws]
TM=[nums[6] for _,nums in draws]
N=len(TM)

# Attributes mappings (2025)
num_to_color={}
for cname,key in [("红波","红"),("蓝波","蓝"),("绿波","绿")]:
    cm=re.search(r">"+cname+"<.*?</td>\s*<td>([\s\S]*?)</td>", sx_html)
    if cm:
        for n in re.findall(r"(\d{1,2})", cm.group(1)):
            num_to_color[int(n)]=key

# Build models function

def last_n(seq,n):
    return seq[-n:] if n<len(seq) else list(seq)

def compute_models(seq):
    N=len(seq)
    freq_full=Counter(seq)
    freq120=Counter(last_n(seq,120))
    freq80=Counter(last_n(seq,80))
    freq50=Counter(last_n(seq,50))
    freq15=Counter(last_n(seq,15))
    last_seen={n:-1 for n in range(1,50)}
    for i,x in enumerate(seq): last_seen[x]=i
    om={n:(N-1-last_seen[n] if last_seen[n]!=-1 else N) for n in range(1,50)}
    alpha=2.0/21.0
    ema={n:0.0 for n in range(1,50)}
    for x in seq:
        for n in range(1,50):
            ema[n]=alpha*(1.0 if n==x else 0.0)+(1.0-alpha)*ema[n]
    trans={i:Counter() for i in range(1,50)}
    for a,b in zip(seq[:-1], seq[1:]): trans[a][b]+=1
    def topk(scores,k=5):
        return [n for n,_ in sorted(scores.items(), key=lambda x:(-x[1], x[0]))[:k]]
    models={}
    # 1 频率平衡
    buckets=defaultdict(list)
    for n in range(1,50): buckets[(num_to_color.get(n,''), n%2)].append(n)
    for bk in buckets: buckets[bk]=sorted(buckets[bk], key=lambda n:(-freq120.get(n,0), n))
    res=[]; keys=sorted(buckets.keys(), key=lambda b: -sum(freq120.get(n,0) for n in buckets[b]))
    while len(res)<5 and any(buckets.values()):
        for bk in keys:
            if buckets[bk]:
                x=buckets[bk].pop(0)
                if x not in res: res.append(x)
                if len(res)>=5: break
    models['频率平衡预测']=res
    # 2 遗漏
    models['遗漏值预测']=[n for n,_ in sorted(om.items(), key=lambda x:(-x[1], x[0]))[:5]]
    # 3 趋势
    models['趋势预测']=topk(ema)
    # 4 关联性
    last=seq[-1]; row=trans.get(last,{})
    cand={n:row.get(n,0)+1e-6 for n in range(1,50) if n!=last}
    models['关联性预测']=topk(cand)
    # 5 聚类
    random.seed(42)
    X={n:[float(trans[n].get(m,0)) for m in range(1,50)] for n in range(1,50)}
    for n in X:
        v=X[n]; norm=math.sqrt(sum(x*x for x in v)) or 1.0
        X[n]=[x/norm for x in v]
    K=5; centers=[X[n] for n in random.sample(range(1,50),K)]
    labels={n:0 for n in range(1,50)}
    for _ in range(8):
        for n in range(1,50):
            sims=[sum(X[n][i]*c[i] for i in range(49)) for c in centers]
            labels[n]=max(range(K), key=lambda j:sims[j])
        new=[]
        for j in range(K):
            mem=[n for n in range(1,50) if labels[n]==j]
            if not mem: new.append(centers[j])
            else:
                acc=[0.0]*49
                for n in mem:
                    vv=X[n]
                    for i in range(49): acc[i]+=vv[i]
                norm=math.sqrt(sum(x*x for x in acc)) or 1.0
                new.append([x/norm for x in acc])
        centers=new
    cluster_scores=defaultdict(float)
    for j in range(K):
        mem=[n for n in range(1,50) if labels[n]==j]
        if mem:
            cluster_scores[j]=sum(freq120.get(n,0) for n in mem)/len(mem)
    best=max(cluster_scores, key=lambda j:cluster_scores[j]) if cluster_scores else 0
    cand=[n for n in range(1,50) if labels[n]==best]
    models['聚类预测']=sorted(cand, key=lambda n:(-freq120.get(n,0), n))[:5]
    # 6 机器学习
    def zscore(d):
        vals=list(d.values()); m=sum(vals)/len(vals); v=sum((x-m)**2 for x in vals)/len(vals); s=math.sqrt(v) or 1.0
        return {k:(d[k]-m)/s for k in d}
    zf=zscore({n:Counter(last_n(seq,50)).get(n,0) for n in range(1,50)})
    zo=zscore(om)
    ml={n:0.6*zf.get(n,0.0)+0.4*zo.get(n,0.0) for n in range(1,50)}
    models['机器学习预测']=topk(ml)
    # 7 频率分析
    weights=[(freq_full,0.4),(freq120,0.25),(freq80,0.2),(Counter(last_n(seq,50)),0.1),(freq15,0.05)]
    sc=defaultdict(float)
    for d,w in weights:
        mx=max(d.values()) if d else 1
        for n in range(1,50): sc[n]+=w*(d.get(n,0)/(mx or 1))
    models['频率分析模型']=topk(sc)
    # 8 统计学
    p=1/49; mu=N*p; sigma=math.sqrt(N*p*(1-p)) or 1.0
    zfull={n:(freq_full.get(n,0)-mu)/sigma for n in range(1,50)}
    models['统计学模型']=topk(zfull)
    # 9 ARIMA-lite
    diffs=[b-a for a,b in zip(seq[:-1], seq[1:])]
    md=sum(diffs[-20:])/min(20,len(diffs)) if diffs else 0
    fc=seq[-1]+md
    sar={}
    for n in range(1,50):
        dist=min(abs(n-fc), 49-abs(n-fc))
        sar[n]=(1/(1+dist))+0.2*Counter(last_n(seq,50)).get(n,0)
    models['ARIMA模型预测']=topk(sar)
    # 10 LSTM适配 ngram
    o3=defaultdict(lambda:Counter())
    for i in range(3,N): o3[(seq[i-3],seq[i-2],seq[i-1])][seq[i]]+=1
    key=(seq[-3],seq[-2],seq[-1]); row3=o3.get(key, {})
    if row3: cand=dict(row3)
    else:
        o2=defaultdict(lambda:Counter())
        for i in range(2,N): o2[(seq[i-2],seq[i-1])][seq[i]]+=1
        cand=dict(o2.get((seq[-2],seq[-1]), {}))
    for n in range(1,50): cand[n]=cand.get(n,0)
    models['LSTM模型适配预测']=[n for n,_ in sorted(cand.items(), key=lambda x:(-x[1], x[0]))[:5]]
    # 11 streak parity
    lp=seq[-1]%2; st=1
    for x in seq[-2::-1]:
        if x%2==lp: st+=1
        else: break
    tp=1-lp if st>=3 else lp
    cand=[n for n in range(1,50) if n%2==tp]
    models['统计检验模型预测']=sorted(cand, key=lambda n:(-om.get(n,0), -Counter(last_n(seq,50)).get(n,0), n))[:5]
    # 12 Bayes
    pm={n:(freq_full.get(n,0)+1)/(N+2) for n in range(1,50)}
    rc={n:Counter(last_n(seq,15)).get(n,0)/max(1,min(15,N)) for n in range(1,50)}
    bay={n:pm[n]+0.5*rc[n] for n in range(1,50)}
    models['贝叶斯后验概率模型预测']=topk(bay)
    # 13 Markov two-stage (quick heuristic using full-year tmsx; may be empty)
    if tmsx_html:
        rows=[]
        for m in re.finditer(r"<tr>\s*<td>(\d{1,3})</td>([\s\S]*?)</tr>", tmsx_html):
            issue=int(m.group(1))
            cells=m.group(2)
            tds=re.findall(r"<td[\s\S]*?>[\s\S]*?</td>", cells)
            idx=None
            for i,td in enumerate(tds):
                if 'tema' in td: idx=i; break
            if idx is None: continue
            zods=["鼠","牛","虎","兔","龙","蛇","马","羊","猴","鸡","狗","猪"]
            if 0<=idx<12 and issue<= (issues[len(seq)-1] if len(seq)<=len(issues) else 9999):
                rows.append((issue, zods[idx]))
        rows.sort(key=lambda x:x[0])
        zx_series=[z for _,z in rows]
        zx_trans=defaultdict(lambda: Counter())
        for a,b in zip(zx_series[:-1], zx_series[1:]): zx_trans[a][b]+=1
        last_issue = issues[len(seq)-1] if len(seq)<=len(issues) else issues[-1]
        last_zx = next((z for iss,z in rows[::-1] if iss==last_issue), '')
        next_zx = zx_trans[last_zx].most_common(1)[0][0] if last_zx and zx_trans[last_zx] else ''
        # map from sx page
        num_to_zx={}
        for m in re.finditer(r"<span class=\"sx\">([\u4e00-\u9fa5]+).*?</span>([\s\S]*?)(?=<span class=\"sx\"|</li>|</ul>)", sx_html):
            name=m.group(1)[:1]
            for n in re.findall(r"(\d{1,2})", m.group(2)):
                num_to_zx[int(n)]=name
        zx_cands=[n for n,z in num_to_zx.items() if z==next_zx]
        models['马尔科夫链高级二阶段模型预测'] = sorted(zx_cands, key=lambda n:(-freq120.get(n,0), -n))[:5]
    else:
        models['马尔科夫链高级二阶段模型预测'] = []
    # 14 dynamic matrix
    l60=last_n(seq,60)
    hc=Counter([x//10 for x in l60]); tc=Counter([x%10 for x in l60])
    ht=sum(hc.values()) or 1; tt=sum(tc.values()) or 1
    dm={}
    for n in range(1,50): dm[n]=(hc.get(n//10,0)/ht)*(tc.get(n%10,0)/tt)
    models['动态概率矩阵模型预测']=topk(dm)
    # 15 RNN-kernel
    Kseq=last_n(seq,12); w=[math.exp(-0.2*(len(Kseq)-1-i)) for i in range(len(Kseq))]
    kr={n:0.0 for n in range(1,50)}
    for i,x in enumerate(Kseq):
        for n in range(1,50):
            d=min(abs(n-x), 49-abs(n-x))
            kr[n]+=w[i]*math.exp(-0.5*(d/3.0)**2)
    models['LSTM神经网络模型预测']=topk(kr)
    # 16 majority color+parity
    cols=[num_to_color.get(x,'') for x in last_n(seq,15)]
    pars=[x%2 for x in last_n(seq,15)]
    cm=Counter(cols).most_common(1)[0][0] if cols else ''
    pmj=Counter(pars).most_common(1)[0][0] if pars else 0
    cand=[n for n in range(1,50) if num_to_color.get(n,'')==cm and n%2==pmj]
    models['逐期解析模型预测']=sorted(cand, key=lambda n:(-Counter(last_n(seq,50)).get(n,0), n))[:5]
    # 17 exact freq
    models['精确频率统计模型预测']=[n for n,_ in Counter(seq).most_common() if n!=seq[-1]][:5]
    # 18 trans matrix
    row=trans.get(seq[-1], {})
    rs={n:row.get(n,0)+1 for n in range(1,50) if n!=seq[-1]}
    models['转移矩阵模型预测']=[n for n,_ in sorted(rs.items(), key=lambda x:(-x[1], x[0]))[:5]]
    # 19 python blend
    def minmax(d):
        if not d: return {}
        vmin=min(d.values()); vmax=max(d.values());
        return {k:(d[k]-vmin)/(vmax-vmin) if vmax>vmin else 0.5 for k in d}
    comb={n:0.5*sc.get(n,0.0)+0.3*minmax(ema).get(n,0.0)+0.2*minmax({k:(freq_full.get(k,0)+1)/(N+2)+0.5*freq15.get(k,0)/max(1,min(15,N)) for k in range(1,50)}).get(n,0.0) for n in range(1,50)}
    models['Python模型预测']=topk(comb)
    # 20 zodiac ratio (skip)
    models['生肖比例模型预测']=[]
    return models

# Backtest last 100 steps within 2025 (use issues ordering)
start_idx = max(0, len(issues) - 100 - 1)
end_idx = len(issues) - 2
model_names = None
hit_counts = defaultdict(int)
total = 0
for idx in range(start_idx, end_idx+1):
    hist_tm = [nums[6] for _,nums in draws[:idx+1]]
    models = compute_models(hist_tm)
    if model_names is None:
        model_names = list(models.keys())
    actual = draws[idx+1][1][6]
    for name, lst in models.items():
        if actual in lst[:5]:
            hit_counts[name] += 1
    total += 1

rates = [(name, (hit_counts.get(name,0), total, hit_counts.get(name,0)/total if total else 0.0)) for name in model_names]
rates.sort(key=lambda x: (-x[1][2], -x[1][0], x[0]))
print('TOTAL_STEPS', total)
for name, (hits, tot, rate) in rates[:5]:
    print(f'{name}: {hits}/{tot} = {rate:.3f}')
