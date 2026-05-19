#!/usr/bin/env python3
"""Parse streaming_profile_*.txt files and extract A/B comparison metrics."""
import os, re, sys, json, glob

def parse_file(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    d = {'path': os.path.basename(path)}

    def find_float(pattern, default=None):
        m = re.search(pattern, text, re.MULTILINE)
        if m: return float(m.group(1))
        return default
    def find_int(pattern, default=None):
        m = re.search(pattern, text, re.MULTILINE)
        if m: return int(m.group(1))
        return default

    # Header
    d['wall_s']       = find_float(r'Wall time:\s+([\d.]+)s')
    d['requested']    = find_int(r'Requested:\s+(\d+)')
    d['completed']    = find_int(r'Completed:\s+(\d+)')
    d['slow']         = find_int(r'Slow path:\s+(\d+)')
    d['fast']         = find_int(r'Fast path:\s+(\d+)')

    # Per-Phase Totals (we capture Total column, last float on the line for each phase)
    def phase_total(name):
        m = re.search(rf'^\s*{name}\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)', text, re.MULTILINE)
        return float(m.group(1)) if m else None
    def phase_avg(name):
        m = re.search(rf'^\s*{name}\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+[\d.]+', text, re.MULTILINE)
        return float(m.group(1)) if m else None
    def phase_max(name):
        m = re.search(rf'^\s*{name}\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+', text, re.MULTILINE)
        return float(m.group(1)) if m else None

    d['region_density_total']    = phase_total('region_density')
    d['hermite_total']           = phase_total('hermite')
    d['dc_solve_total']          = phase_total('dc_solve')
    d['mesh_gen_total']          = phase_total('mesh_gen')
    d['seam_pass_total']         = phase_total('seam_pass')
    d['seam_pass_avg']           = phase_avg('seam_pass')
    d['seam_pass_max']           = phase_max('seam_pass')
    d['store_read_wait_total']   = phase_total('store_read_wait')
    d['store_read_wait_avg']     = phase_avg('store_read_wait')
    d['store_read_wait_max']     = phase_max('store_read_wait')
    d['store_write_wait_total']  = phase_total('store_write_wait')
    d['store_write_wait_avg']    = phase_avg('store_write_wait')
    d['store_write_wait_max']    = phase_max('store_write_wait')
    d['total_total']             = phase_total('TOTAL')

    # Seam pass breakdown
    d['seam_quad_gen_total']     = phase_total('quad_gen')
    d['seam_mesh_retrieve_total']= phase_total('mesh_retrieve')
    d['seam_convert_total']      = phase_total('convert')
    m = re.search(r'Candidates:\s+tried=(\d+)\s+sent=(\d+)', text)
    if m:
        d['seam_tried'] = int(m.group(1)); d['seam_sent'] = int(m.group(2))
        d['seam_hit_rate'] = d['seam_sent']/d['seam_tried']*100 if d['seam_tried'] else 0

    # Latency
    d['lat_count'] = find_int(r'^\s*Count:\s+(\d+)')
    d['lat_avg']   = find_float(r'^\s*Avg:\s+([\d.]+)\s+ms')
    d['lat_p95']   = find_float(r'^\s*P95:\s+([\d.]+)\s+ms')
    d['lat_max']   = find_float(r'^\s*Max:\s+([\d.]+)\s+ms')

    # Wall time budget
    d['ue_processresults_ms'] = find_float(r'UE main thread in ProcessResults:\s+([\d.]+)\s+ms')
    d['ue_applymesh_ms']      = find_float(r'ApplyMeshData \(executed\):\s+([\d.]+)\s+ms')
    d['ue_createmesh_ms']     = find_float(r'CreateMeshSection \(slow\):\s+([\d.]+)\s+ms')

    # UE streaming
    d['ue_wall_ms']          = find_float(r'Wall time \(UE\):\s+([\d.]+)ms')
    d['chunks_spawned']      = find_int(r'Chunks spawned \(new actors\):\s+(\d+)')
    d['chunks_updated']      = find_int(r'Chunks updated \(seam/mesh\):\s+(\d+)')
    d['procres_avg_ms']      = find_float(r'^\s*Avg/frame:\s+([\d.]+)ms')
    d['procres_max_ms']      = find_float(r'^\s*Max single frame:\s+([\d.]+)ms')
    d['stall_frames_pct']    = find_float(r'Stall frames:\s+\d+\s+\(([\d.]+)%\)')
    d['applies_executed']    = find_int(r'Applies executed:\s+(\d+)')
    d['applies_skipped']     = find_int(r'Applies skipped \(hash hit\):\s+(\d+)')

    # Tag extraction from filename
    m = re.search(r'streaming_profile_\d{4}-\d{2}-\d{2}_\d{6}(?:_(.+))?\.txt', d['path'])
    if m:
        d['tag'] = m.group(1) or 'untagged'
    return d

def summarize(files, label):
    rows = [parse_file(f) for f in files]
    # Group by tag
    print(f"\n{'='*70}\n {label}  ({len(rows)} files)\n{'='*70}")
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r['tag'],[]).append(r)
    for tag, rs in sorted(by_tag.items()):
        print(f"\n-- tag={tag}  (n={len(rs)})")
        keys = ['wall_s','requested','completed','seam_pass_total','seam_pass_avg','seam_pass_max',
                'store_read_wait_avg','store_read_wait_max','store_write_wait_avg','store_write_wait_max',
                'seam_tried','seam_sent','seam_hit_rate','lat_avg','lat_p95','lat_max',
                'ue_processresults_ms','procres_max_ms','stall_frames_pct','applies_executed','applies_skipped']
        for k in keys:
            vals = [r.get(k) for r in rs if r.get(k) is not None]
            if not vals: continue
            avg = sum(vals)/len(vals)
            mx  = max(vals)
            print(f"  {k:30s}  avg={avg:10.2f}  max={mx:10.2f}  n={len(vals)}")
    return rows

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: parse.py <dir> [label]")
        sys.exit(1)
    d = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else d
    files = sorted(glob.glob(os.path.join(d,'streaming_profile_*.txt')))
    summarize(files, label)
