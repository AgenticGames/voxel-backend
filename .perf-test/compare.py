#!/usr/bin/env python3
"""Compact A vs B comparison focused on changed metrics."""
import os, re, sys, glob

METRICS = [
    ('wall_s',                  'Wall time (s)',          'lower'),
    ('seam_pass_total',         'Seam pass total (ms)',   'lower'),
    ('seam_pass_avg',           'Seam pass avg (ms)',     'lower'),
    ('seam_pass_max',           'Seam pass max (ms)',     'lower'),
    ('seam_mesh_retrieve_total','Seam retrieve total',    'lower'),
    ('seam_convert_total',      'Seam convert total',     'lower'),
    ('seam_sent_pct',           'Seam sent % (lower=skip win)', 'lower'),
    ('store_write_wait_avg',    'Store write wait avg',   'lower'),
    ('store_write_wait_max',    'Store write wait max',   'lower'),
    ('store_read_wait_avg',     'Store read wait avg',    'lower'),
    ('store_read_wait_max',     'Store read wait max',    'lower'),
    ('lat_avg',                 'Req-to-Result avg (ms)', 'lower'),
    ('lat_p95',                 'Req-to-Result P95 (ms)', 'lower'),
    ('lat_max',                 'Req-to-Result max (ms)', 'lower'),
    ('ue_processresults_ms',    'UE ProcessResults (ms)', 'lower'),
    ('procres_max_ms',          'ProcRes max frame (ms)', 'lower'),
    ('stall_frames_pct',        'Stall frames %',         'lower'),
]

def parse_file(path):
    text = open(path, 'r', encoding='utf-8', errors='replace').read()
    d = {'path': os.path.basename(path)}
    def f(pat, default=None, cast=float):
        m = re.search(pat, text, re.MULTILINE)
        return cast(m.group(1)) if m else default

    d['wall_s']      = f(r'Wall time:\s+([\d.]+)s')
    d['requested']   = f(r'Requested:\s+(\d+)', cast=int)

    def phase(name, col):  # col 1=min 2=avg 3=max 4=total
        m = re.search(rf'^\s*{name}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', text, re.MULTILINE)
        return float(m.group(col)) if m else None

    for n in ['region_density','hermite','seam_pass','store_read_wait','store_write_wait']:
        d[f'{n}_avg']   = phase(n, 2)
        d[f'{n}_max']   = phase(n, 3)
        d[f'{n}_total'] = phase(n, 4)
    d['seam_mesh_retrieve_total'] = phase('mesh_retrieve', 4)
    d['seam_convert_total']       = phase('convert', 4)

    m = re.search(r'Candidates:\s+tried=(\d+)\s+sent=(\d+)', text)
    if m:
        t,s = int(m.group(1)), int(m.group(2))
        d['seam_tried']=t; d['seam_sent']=s
        d['seam_sent_pct'] = s/t*100 if t else None

    d['lat_avg'] = f(r'Req.{0,20}Latency.*?Avg:\s+([\d.]+)', cast=float)
    # simpler
    d['lat_avg'] = None; d['lat_p95']=None; d['lat_max']=None
    m = re.search(r'Request-to-Result Latency.*?Avg:\s+([\d.]+).*?P95:\s+([\d.]+).*?Max:\s+([\d.]+)', text, re.S)
    if m:
        d['lat_avg']=float(m.group(1)); d['lat_p95']=float(m.group(2)); d['lat_max']=float(m.group(3))

    d['ue_processresults_ms'] = f(r'UE main thread in ProcessResults:\s+([\d.]+)')
    d['procres_max_ms']       = f(r'Max single frame:\s+([\d.]+)ms')
    d['stall_frames_pct']     = f(r'Stall frames:\s+\d+\s+\(([\d.]+)%\)')
    d['ue_wall_ms']           = f(r'Wall time \(UE\):\s+([\d.]+)ms')

    # Tag
    m = re.search(r'streaming_profile_\d{4}-\d{2}-\d{2}_\d{6}(?:_(.+))?\.txt', d['path'])
    d['tag'] = m.group(1) if m and m.group(1) else 'untagged'
    # Group tag (strip burst numbers)
    g = re.sub(r'burst_\d+(_.+)?', lambda m: 'burst'+(m.group(1) or ''), d['tag'])
    d['gtag'] = g
    return d

def collect(d):
    by = {}
    for f in sorted(glob.glob(os.path.join(d,'streaming_profile_*.txt'))):
        r = parse_file(f)
        by.setdefault(r['gtag'], []).append(r)
    return by

def stats(rows, key):
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    if not vals: return None
    return {'avg': sum(vals)/len(vals), 'max': max(vals), 'min': min(vals), 'n': len(vals)}

def compare(a_dir, b_dir):
    A = collect(a_dir); B = collect(b_dir)
    # Align by group tag
    tags = sorted(set(A.keys()) | set(B.keys()))
    print(f"{'='*96}")
    print(f"  A vs B COMPARISON")
    print(f"  A: {a_dir}  ({sum(len(v) for v in A.values())} files)")
    print(f"  B: {b_dir}  ({sum(len(v) for v in B.values())} files)")
    print(f"{'='*96}")
    for tag in tags:
        arows = A.get(tag, [])
        brows = B.get(tag, [])
        print(f"\n--- tag='{tag}'    A:n={len(arows)}  B:n={len(brows)} ---")
        print(f"  {'Metric':<35s} {'A avg':>10s} {'B avg':>10s} {'delta':>10s} {'pct':>7s}")
        for key, label, direction in METRICS:
            a = stats(arows, key); b = stats(brows, key)
            if not a or not b: continue
            da = a['avg']; db = b['avg']
            delta = db - da
            pct = (delta/da*100) if da else 0
            marker = ''
            if direction == 'lower':
                if pct < -5: marker = ' [better]'
                elif pct > 5: marker = ' [worse]'
            print(f"  {label:<35s} {da:>10.2f} {db:>10.2f} {delta:>+10.2f} {pct:>+6.1f}%{marker}")

if __name__ == '__main__':
    if len(sys.argv)<3:
        print("Usage: compare.py A_dir B_dir"); sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
