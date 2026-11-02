"""Persistence-only adapters; scoring, vectors and selection remain untouched."""
import json
import sqlite3
from pathlib import Path

def validate_existing(directory):
    root=Path(directory)
    shapes={'rgec_states.json':list,'evidence_plan.json':list,'candidate_universe.json':list,
        'query_graphs.json':list,'unrollings.json':list,'retrieval_records.json':dict,'qa_records.json':dict}
    for name,kind in shapes.items():
        p=root/name
        if p.exists() and not isinstance(json.loads(p.read_text()),kind):raise RuntimeError('Invalid checkpoint schema; preserved: '+str(p))
    p=root/'soft_embeddings_v1.sqlite'
    if p.exists():
        connection=sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            if connection.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise RuntimeError('Embedding SQLite integrity failed; database preserved')
        finally:connection.close()

def persist_vectors(cache,rows):
    if not rows:return
    connection=sqlite3.connect(str(cache.cache_path),timeout=30)
    try:
        connection.execute('PRAGMA synchronous=FULL')
        with connection:
            connection.executemany('INSERT OR REPLACE INTO embeddings(text, dim, vector) VALUES (?, ?, ?)',rows)
    finally:connection.close()
