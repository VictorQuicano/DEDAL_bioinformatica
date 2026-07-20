# coding=utf-8
"""Download and parse Pfam seed alignments (Stockholm format).

Pfam was retired as a standalone site in 2022 and now lives inside InterPro,
so `pfam.xfam.org` no longer resolves. The InterPro REST API serves the same
seed alignments.
"""

import os
import re

import requests

SEED_URL = ('https://www.ebi.ac.uk/interpro/api/entry/pfam/'
            '{acc}/?annotation=alignment:seed')

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

# Stockholm sequence rows look like "SEQID/start-end   GAPPED-SEQUENCE".
ROW_RE = re.compile(r'^(\S+/\d+-\d+)\s+(\S+)$')

# The 18 families used in the benchmark. The first six carry the low-identity
# pairs (PID < 0.1) that the comparison actually hinges on; the last four keep
# the high-identity end populated so both curves converge on the right.
FAMILIES = [
    'PF00104', 'PF00583', 'PF00043', 'PF00041', 'PF07686', 'PF00089',
    'PF00595', 'PF01565', 'PF00013', 'PF00027', 'PF00076', 'PF00072',
    'PF00010', 'PF00085', 'PF00018', 'PF00046', 'PF00014', 'PF00105',
]


def fetch_seed(acc, cache_dir=CACHE_DIR):
    """Return the raw Stockholm text of a family's seed alignment.

    Reads from the on-disk cache when available, so the rest of the pipeline
    behaves identically whether the files were downloaded here or by hand.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'{acc}.sto')

    if os.path.exists(path):
        with open(path, encoding='latin-1') as f:
            return f.read()

    response = requests.get(SEED_URL.format(acc=acc), timeout=60)
    response.raise_for_status()
    # The API sets Content-Encoding: gzip at the transport level, so requests
    # has already decompressed this. Calling gzip.decompress here would fail.
    text = response.text

    with open(path, 'w', encoding='latin-1') as f:
        f.write(text)
    return text


def parse_stockholm(text):
    """Parse Stockholm text into an ordered {row_name: gapped_sequence} dict."""
    rows = {}
    for line in text.splitlines():
        line = line.rstrip('\n')
        # Markup lines (#=GC, #=GR, #=GS, #=GF) would otherwise match ROW_RE.
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        match = ROW_RE.match(line)
        if match is None:
            continue
        name, chunk = match.groups()
        # Stockholm allows an alignment to be split across several blocks.
        rows[name] = rows.get(name, '') + chunk

    lengths = {len(v) for v in rows.values()}
    if len(lengths) > 1:
        raise ValueError(f'Inconsistent row lengths in alignment: {lengths}')
    return rows


def ungapped(gapped):
    """Strip gap characters, keeping every residue.

    Lowercase characters mark insert columns in Pfam's convention: they are
    still residues of the sequence, so they must be kept (and uppercased) or
    every downstream index shifts.
    """
    return ''.join(c for c in gapped if c.isalpha()).upper()


def load_family(acc, cache_dir=CACHE_DIR):
    """Fetch and parse one family. Returns {row_name: gapped_sequence}."""
    return parse_stockholm(fetch_seed(acc, cache_dir))
