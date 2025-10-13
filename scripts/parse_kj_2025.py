# -*- coding: utf-8 -*-
import re
import csv
from bs4 import BeautifulSoup
from pathlib import Path

INPUT_HTML = Path('/workspace/data/kj_2025.html')
OUTPUT_CSV = Path('/workspace/data/kj_2025_parsed.csv')


def extract_issue_number(title_txt: str) -> int:
    # title like: "六合彩开奖记录 2025年10月12日 第285期"
    m = re.search(r"第\s*(\d{1,3})\s*期", title_txt)
    if not m:
        return None
    return int(m.group(1))


def parse_one_block(title_div, box_div):
    issue = extract_issue_number(title_div.get_text(" ", strip=True))
    # In each .kj-box > ul > li, last li after class kj-jia is 第7码(特码)
    nums = []
    attrs = []
    for li in box_div.select('ul > li'):
        dt = li.select_one('dt')
        dd_all = li.select('dd')
        if not dt:
            continue
        num_txt = (dt.get_text(strip=True) or '').strip()
        if not num_txt:
            continue
        nums.append(num_txt)
        # dd[0]: 生肖/五行, dd[1]: 波色/大小 (display:none), dd[2]: 单双/合单双, dd[3]: 家禽野兽/总和单双
        sx = wx = bs = dx = ds = hds = jqys = zshds = None
        if len(dd_all) >= 1:
            parts = [t.get_text(strip=True) for t in dd_all[0].contents if getattr(t, 'get_text', None)]
            # dd_all[0] is like: "猴 / 木"; more robustly split by '/'
            text0 = dd_all[0].get_text('/', strip=True)
            sx_wx = [p.strip() for p in text0.split('/')]
            if len(sx_wx) >= 1:
                sx = sx_wx[0]
            if len(sx_wx) >= 2:
                wx = sx_wx[1]
        if len(dd_all) >= 2:
            text1 = dd_all[1].get_text('/', strip=True)
            bs_dx = [p.strip() for p in text1.split('/')]
            if len(bs_dx) >= 1:
                bs = bs_dx[0]
            if len(bs_dx) >= 2:
                dx = bs_dx[1]
        if len(dd_all) >= 3:
            text2 = dd_all[2].get_text('/', strip=True)
            ds_hds = [p.strip() for p in text2.split('/')]
            if len(ds_hds) >= 1:
                ds = ds_hds[0]
            if len(ds_hds) >= 2:
                hds = ds_hds[1]
        if len(dd_all) >= 4:
            text3 = dd_all[3].get_text('/', strip=True)
            jqys_zshds = [p.strip() for p in text3.split('/')]
            if len(jqys_zshds) >= 1:
                jqys = jqys_zshds[0]
            if len(jqys_zshds) >= 2:
                zshds = jqys_zshds[1]
        attrs.append({
            'sx': sx, 'wx': wx, 'bs': bs, 'dx': dx,
            'ds': ds, 'hds': hds, 'jqys': jqys, 'zshds': zshds
        })
    # 特码在 'kj-jia' 之后的下一个 li 的 dt
    # 页面结构：前6个正码，然后1个 class=kj-jia 的空位，然后最后一个是特码
    # 过滤掉 class=kj-jia 的 li
    lis = [li for li in box_div.select('ul > li') if 'kj-jia' not in li.get('class', [])]
    if not lis:
        return None
    # 通常第7个(索引6)是特码
    tema_li = lis[-1]
    tema_dt = tema_li.select_one('dt')
    tema_dds = tema_li.select('dd')
    tema_num = int(tema_dt.get_text(strip=True)) if tema_dt else None
    tema_sx = tema_wx = tema_bs = tema_dx = tema_ds = tema_hds = tema_jqys = tema_zshds = None
    if len(tema_dds) >= 1:
        text0 = tema_dds[0].get_text('/', strip=True)
        sx_wx = [p.strip() for p in text0.split('/')]
        if sx_wx:
            tema_sx = sx_wx[0]
        if len(sx_wx) > 1:
            tema_wx = sx_wx[1]
    if len(tema_dds) >= 2:
        text1 = tema_dds[1].get_text('/', strip=True)
        bs_dx = [p.strip() for p in text1.split('/')]
        if bs_dx:
            tema_bs = bs_dx[0]
        if len(bs_dx) > 1:
            tema_dx = bs_dx[1]
    if len(tema_dds) >= 3:
        text2 = tema_dds[2].get_text('/', strip=True)
        ds_hds = [p.strip() for p in text2.split('/')]
        if ds_hds:
            tema_ds = ds_hds[0]
        if len(ds_hds) > 1:
            tema_hds = ds_hds[1]
    if len(tema_dds) >= 4:
        text3 = tema_dds[3].get_text('/', strip=True)
        jqys_zshds = [p.strip() for p in text3.split('/')]
        if jqys_zshds:
            tema_jqys = jqys_zshds[0]
        if len(jqys_zshds) > 1:
            tema_zshds = jqys_zshds[1]

    return {
        'issue': issue,
        'tema': tema_num,
        'tema_sx': tema_sx,
        'tema_wx': tema_wx,
        'tema_bs': tema_bs,
        'tema_dx': tema_dx,
        'tema_ds': tema_ds,
        'tema_hds': tema_hds,
        'tema_jqys': tema_jqys,
        'tema_zshds': tema_zshds,
    }


def main():
    html = INPUT_HTML.read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    for title_div in soup.select('.kj-tit'):
        # next sibling is the kj-box for that period
        box_div = title_div.find_next_sibling('div', class_='kj-box')
        if not box_div:
            continue
        rec = parse_one_block(title_div, box_div)
        if not rec or not rec['issue']:
            continue
        rows.append(rec)
    if not rows:
        raise SystemExit('No rows parsed')
    # Keep only issues for 2025; page is already 2025. Sort by issue asc
    rows = sorted(rows, key=lambda r: r['issue'])
    # We only need 001..285 according to user
    rows = [r for r in rows if 1 <= int(r['issue']) <= 285]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'issue','tema','tema_sx','tema_wx','tema_bs','tema_dx','tema_ds','tema_hds','tema_jqys','tema_zshds'
            ]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f'Parsed {len(rows)} issues saved to {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
