#!/usr/bin/env python3
"""
Collect MAE/MRE/R2 from logs under ./output.
Handles both numeric folds (e.g., "Fold 2") and named folds like
"Fold mf.chemsys.k0_outer" (split=chemsys, fold=0).

Expected tree (recursive):
output/
  split/
    alignn_*.txt
    cgcnn_*.txt
    ...
  ood/
    *.txt
  finetune/
    *.txt

CSV written to ROOT/metrics_summary.csv (or --outfile).
"""

import re, ast, csv
from pathlib import Path

FOLD_BLOCK_RE = re.compile(
    r'(Validation|Test)\s+Results\s*\(Fold\s*([^)]+)\)\s*:\s*(\[[^\]]*\])',
    re.IGNORECASE
)
RESULT_LINE_RE = re.compile(
    r'^\s*(.+?)\s+(Validation|Test)\s+Result-\s*(.+)$',
    re.IGNORECASE | re.MULTILINE
)
FOLD_NAME_RE = re.compile(
    r'Fold\s+(\d+)/[^\s:]+:\s*([^\s:]+)',
    re.IGNORECASE
)
METRIC_PAIR_RE = re.compile(r'([A-Za-z²]+)\s*:\s*([-+0-9.eE]+)')

def infer_method_from_filename(fname: str) -> str:
    name = Path(fname).stem
    parts = name.split('_')
    # strip trailing numeric job id tokens
    while parts and parts[-1].isdigit():
        parts.pop()
    return '_'.join(parts) if parts else name

def parse_fold_token(token: str):
    """
    token can be:
      - '2'                          -> (None, 2, '2')
      - 'mf.chemsys.k0_outer'        -> ('chemsys', 0, token)
      - 'mf.sgnum.k3_outer'          -> ('sgnum', 3, token)
      - 'k4_inner'                   -> (None, 4, token)
    Returns (split_name, fold_idx, fold_token_str)
    """
    t = token.strip()
    # numeric only
    if t.isdigit():
        return (None, int(t), t)

    # try to find pattern '.<split>.k<idx>_'
    m = re.search(r'\.([a-zA-Z0-9]+)\.k(\d+)_', t)
    if m:
        split_name = m.group(1)
        fold_idx = int(m.group(2))
        return (split_name, fold_idx, t)

    # try simpler 'k<idx>_' anywhere
    m = re.search(r'k(\d+)_', t)
    if m:
        fold_idx = int(m.group(1))
        return (None, fold_idx, t)

    # fallback: treat token as split name if nothing else
    return (t, None, t)

def parse_metrics_blocks(text: str):
    """
    Yields tuples: (phase, split_name, fold_idx, fold_token, metrics_dict)
    metrics_dict keys: loss, mae, mre, r2
    Keeps the last occurrence if multiple blocks for the same (phase, split_name/fold_idx/fold_token).
    """
    results = {}
    for m in FOLD_BLOCK_RE.finditer(text):
        phase = m.group(1).lower()  # 'validation' or 'test'
        token = m.group(2)
        payload = m.group(3)

        split_name, fold_idx, fold_token = parse_fold_token(token)

        try:
            lst = ast.literal_eval(payload)
            if isinstance(lst, list) and lst and isinstance(lst[0], dict):
                d = lst[0]
                rec = {
                    'loss': d.get('test_loss', d.get('loss')),
                    'mae':  d.get('test_mae',  d.get('mae')),
                    'mre':  d.get('test_mre',  d.get('mre')),
                    'r2':   d.get('test_r2',   d.get('r2')),
                }
                key = (phase, split_name, fold_idx, fold_token, None)
                results[key] = rec
        except Exception:
            continue
    return results

def normalize_metric_key(key: str) -> str:
    return key.upper().replace('²', '2')

def build_fold_name_map(text: str):
    mapping = {}
    for m in FOLD_NAME_RE.finditer(text):
        fold_idx = int(m.group(1))
        name = m.group(2).strip()
        if name and name not in mapping:
            mapping[name] = fold_idx
    return mapping

def deduce_prefix_info(prefix: str, fold_map):
    parts = prefix.split()
    split_name = None
    fold_idx = None
    fold_token = None
    method_name = None
    if not parts:
        return split_name, fold_idx, fold_token, method_name

    if parts[0].isdigit():
        fold_idx = int(parts[0])
        fold_token = parts[0]
        if len(parts) > 1:
            method_name = parts[1]
    else:
        token = parts[0]
        # Support tokens like "mf.periodictablegroups.k0_outer"
        # by normalizing them through parse_fold_token.
        p_split, p_idx, p_token = parse_fold_token(token)
        split_name = p_split if p_split is not None else token
        fold_token = p_token if p_token is not None else token
        if p_idx is not None:
            fold_idx = p_idx
        else:
            fold_idx = fold_map.get(token)
        if len(parts) > 1:
            method_name = parts[1]
    return split_name, fold_idx, fold_token, method_name

def parse_simple_result_lines(text: str):
    """
    Parse lines like:
      "1 linear_regression Validation Result- MAE: 0.6, MSE: 0.7, MRE: 0.3, R²: 0.33"
    Returns dict keyed similarly to parse_metrics_blocks (phase, split_name, fold_idx, fold_token).
    """
    results = {}
    fold_map = build_fold_name_map(text)
    for m in RESULT_LINE_RE.finditer(text):
        prefix = m.group(1).strip()
        phase = m.group(2).strip().lower()
        metrics_blob = m.group(3)

        split_name, fold_idx, fold_token, method_name = deduce_prefix_info(prefix, fold_map)
        if fold_token is None:
            fold_token = prefix

        rec = {}
        for k, val in METRIC_PAIR_RE.findall(metrics_blob):
            key = normalize_metric_key(k)
            try:
                num = float(val)
            except ValueError:
                continue
            if key == 'MAE':
                rec['mae'] = num
            elif key == 'MRE':
                rec['mre'] = num
            elif key in ('R2', 'R'):
                rec['r2'] = num
            elif key in ('MSE', 'LOSS'):
                rec['loss'] = num
        if not rec:
            continue
        rec['_method'] = method_name
        key = (phase, split_name, fold_idx, fold_token, method_name)
        results[key] = rec
    return results

def main(root="./output", outfile=None):
    root_p = Path(root)
    if not root_p.exists():
        raise SystemExit(f"Root not found: {root}")
    rows = []

    # Consider both .txt and .log files so traditional-ML logs are captured.
    for txt in root_p.rglob("*"):
        if txt.suffix not in (".txt", ".log"):
            continue
        split_dir = txt.parent.name  # e.g., 'split', 'ood', 'finetune'
        method_default = infer_method_from_filename(txt.name)

        try:
            text = txt.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        blocks = parse_metrics_blocks(text)
        simple_blocks = parse_simple_result_lines(text)
        for key, rec in simple_blocks.items():
            blocks.setdefault(key, rec)
        if not blocks:
            continue

        # aggregate rows
        row_index = {}
        def ensure_row(split_name, fold_idx, fold_token, method_override):
            row_method = method_override or method_default
            key = (split_dir, row_method, split_name, fold_idx, fold_token)
            if key not in row_index:
                row = {
                    'split_dir': split_dir,
                    'split_name': split_name,
                    'method': row_method,
                    'fold_idx': fold_idx,
                    'fold_token': fold_token,
                    'val_mae': None, 'val_mre': None, 'val_r2': None, 'val_loss': None,
                    'test_mae': None, 'test_mre': None, 'test_r2': None, 'test_loss': None,
                    'file': str(txt)
                }
                row_index[key] = row
                rows.append(row)
            return row_index[key]

        for (phase, split_name, fold_idx, fold_token, method_tag), rec in blocks.items():
            row = ensure_row(split_name, fold_idx, fold_token, rec.get('_method', method_tag))
            if phase == 'validation':
                row['val_mae']  = rec.get('mae')
                row['val_mre']  = rec.get('mre')
                row['val_r2']   = rec.get('r2')
                row['val_loss'] = rec.get('loss')
            else:
                row['test_mae']  = rec.get('mae')
                row['test_mre']  = rec.get('mre')
                row['test_r2']   = rec.get('r2')
                row['test_loss'] = rec.get('loss')

    # Sort for readability
    rows.sort(key=lambda r: (
        r['split_dir'],
        r['split_name'] or "",
        r['method'],
        (r['fold_idx'] if r['fold_idx'] is not None else 10**9),
        r['fold_token']
    ))

    out_csv = Path(outfile) if outfile else (root_p / "metrics_summary.csv")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            'split_dir','split_name','method','fold_idx','fold_token',
            'val_mae','val_mre','val_r2',
            'test_mae','test_mre','test_r2',
            'val_loss','test_loss','file'
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./output")
    ap.add_argument("--outfile", type=str, default=None)
    args = ap.parse_args()
    main(root=args.root, outfile=args.outfile)
