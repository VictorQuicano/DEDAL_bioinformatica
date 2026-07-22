import os
import re

import requests

SEED_URL = ('https://www.ebi.ac.uk/interpro/api/entry/pfam/'
            '{acc}/?annotation=alignment:seed')

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

ROW_RE = re.compile(r'^(\S+/\d+-\d+)\s+(\S+)$')

FAMILIES = [
    'PF00104', 'PF00583', 'PF00043', 'PF00041', 'PF07686', 'PF00089',
    'PF00595', 'PF01565', 'PF00013', 'PF00027', 'PF00076', 'PF00072',
    'PF00010', 'PF00085', 'PF00018', 'PF00046', 'PF00014', 'PF00105',
]


def fetch_seed(acc, cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'{acc}.sto')

    if os.path.exists(path):
        with open(path, encoding='latin-1') as f:
            return f.read()

    response = requests.get(SEED_URL.format(acc=acc), timeout=60)
    response.raise_for_status()
    text = response.text

    with open(path, 'w', encoding='latin-1') as f:
        f.write(text)
    return text


def parse_stockholm(text):
    rows = {}
    for line in text.splitlines():
        line = line.rstrip('\n')
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        match = ROW_RE.match(line)
        if match is None:
            continue
        name, chunk = match.groups()
        rows[name] = rows.get(name, '') + chunk

    lengths = {len(v) for v in rows.values()}
    if len(lengths) > 1:
        raise ValueError(f'Inconsistent row lengths in alignment: {lengths}')
    return rows


def ungapped(gapped):
    return ''.join(c for c in gapped if c.isalpha()).upper()


def load_family(acc, cache_dir=CACHE_DIR):
    return parse_stockholm(fetch_seed(acc, cache_dir))
