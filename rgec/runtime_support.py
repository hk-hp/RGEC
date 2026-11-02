"""Run-local isolation and read-only offline artifact discovery; no model calls."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

LLM_NAME = "meta-llama/Llama-3.3-70B-Instruct"
EMBEDDING_PATH = "nvidia/NV-Embed-v2"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    # Complete serialization before touching disk. Unique temporary files avoid
    # collisions; fsync before replace preserves the previous valid checkpoint
    # on ENOSPC, quota, serialization failure, or SIGKILL during writing.
    blob=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode('utf8')
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(blob);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if os.path.exists(name):os.unlink(name)


def find_offline(root, explicit=None):
    root = Path(root).resolve()
    openie_name = "openie_results_ner_" + LLM_NAME.replace("/", "_") + ".json"
    tag = LLM_NAME.replace("/", "_") + "_" + EMBEDDING_PATH.replace("/", "_")
    if explicit:
        candidates = [Path(explicit).resolve()]
    else:
        candidates = []
        for name in ("strict_paper_reproduction_musique", "strict_paper_reproduction"):
            for p in (root / name).rglob(openie_name):
                if (p.parent / tag / "graph.pickle").is_file():
                    candidates.append(p.parent.resolve())
        candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one offline source; set SAVE_DIR. Found: {candidates}")
    source = candidates[0]
    relative = [openie_name, f"{tag}/graph.pickle"]
    relative += [f"{tag}/{kind}_embeddings/vdb_{kind}.parquet"
                 for kind in ("chunk", "entity", "fact")]
    files = {name: source / name for name in relative}
    for name, p in files.items():
        if not p.is_file():
            raise RuntimeError(f"Missing offline artifact: {p}")
    return source, files


def stat_snapshot(files):
    return {name: [p.stat().st_size, p.stat().st_mtime_ns]
            for name, p in files.items()}


def link_offline_view(source, files, run_dir):
    """Link only explicitly whitelisted offline files; never historical caches."""
    run_dir = Path(run_dir).resolve()
    source = Path(source).resolve()
    if run_dir == source or source in run_dir.parents or run_dir in source.parents:
        raise RuntimeError("Experiment output and offline source must be separate trees")
    view = run_dir / "offline_view"
    view.mkdir(parents=True, exist_ok=True)
    for name, original in files.items():
        target = view / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != original.resolve():
                raise RuntimeError(f"Offline link points elsewhere: {target}")
        elif target.exists():
            raise RuntimeError(f"Expected a frozen offline link, got regular file: {target}")
        else:
            target.symlink_to(original.resolve())
    return view


def establish_manifest(run_dir, signature):
    """Permit only own, hash-matched resume; never adopt an existing cache tree."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise RuntimeError("Resume identity mismatch. Choose a NEW RUN_DIR; do not delete the manifest")
        return True
    allowed = {"vllm.log", "run.log", "launcher.log"}
    unexpected = [p.name for p in run_dir.iterdir() if p.name not in allowed]
    if unexpected:
        raise RuntimeError(f"Unowned output directory has existing files: {unexpected}")
    atomic_json(manifest, signature)
    return False


def check_data(root, files):
    """Fail before GPUs are reserved if corpus/OpenIE/store alignment is wrong."""
    import pandas as pd
    root = Path(root)
    samples = json.loads((Path(__file__).parent / 'questions100.json').read_text())
    corpus = json.loads((root / "reproduce/dataset/musique_corpus.json").read_text())
    openie_path = next(p for name, p in files.items() if name.endswith(".json"))
    openie = json.loads(openie_path.read_text())["docs"]
    chunk_path = next(p for name, p in files.items() if name.endswith("vdb_chunk.parquet"))
    chunks = pd.read_parquet(chunk_path, columns=["hash_id", "content"])
    if (len(samples), len(corpus), len(openie), len(chunks)) != (100, 11656, 11656, 11656):
        raise RuntimeError("Expected 100 questions and 11656 corpus/OpenIE/chunk rows")
    if len(set(chunks["hash_id"])) != 11656:
        raise RuntimeError("Duplicate chunk IDs")
    if len(set(chunks["content"])) != 11656:
        raise RuntimeError("Duplicate passage text: RGEC content-to-ID mapping would not be one-to-one")
    for i, (text, row) in enumerate(zip(chunks["content"], openie)):
        if text != row.get("passage"):
            raise RuntimeError(f"Fact-to-passage positional mismatch at passage_id={i}; stop before spending API calls")
    corpus_texts = {f"{r['title']}\n{r.get('text', r.get('paragraph_text', ''))}" for r in corpus}
    if corpus_texts != set(chunks["content"]):
        raise RuntimeError("Corpus differs from frozen chunk store")
    return samples, corpus, openie
