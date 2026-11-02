# Relation-Guided Evidence Compilation (RGEC)

Main-method source for **Relation-Guided Evidence Compilation for GraphRAG under
a Fixed Passage Budget**. This package is based on the source code used for the
completed server experiments and has been organized for anonymous source review.

## Method overview

RGEC organizes retrieved evidence around the relations required by a question,
then prioritizes the original source passages within a fixed reading budget.

The question is represented as a directed query graph. Its edges are relation
slots, while shared nodes express dependencies between requirements. The method
then proceeds through three stages:

1. **Query-conditioned candidate acquisition.** Full-question semantic retrieval
   finds candidate facts. Recognition filtering and dense passage retrieval
   initialize entity and passage seeds. Query-relevant graph edges receive a
   bounded weight boost, and graph propagation produces a passage ranking.
2. **Source-grounded evidence compilation.** Facts are ordered by their semantic
   relation alignment to the query slots. An LLM proposes fact-to-slot assignments
   using the supplied source identifiers. Citation and local entity-connection
   checks retain usable assignments as a source-cited evidence plan.
3. **Budgeted context assembly.** The plan's source passages receive priority in
   plan order. Repeated passages are deduplicated, and remaining positions are
   filled from the candidate ranking. The Reader answers from the resulting
   original passages; the default main-experiment budget is five passages.

## Code organization

| Component | Source |
|---|---|
| RGEC method class | `rgec/method.py` |
| Query graph and decomposition prompts | `rgec/query_inputs.py` |
| Candidate acquisition, relation alignment, evidence validation and context assembly | `rgec/evidence_compiler.py` |
| Dataset adapters | `rgec/data_io.py` |
| Experiment execution and evaluation | `rgec/experiment_runtime.py` |
| Configurable entry point | `run_method.py` |
| Upstream framework dependencies | `src/hipporag/` |

This source-only package does not include datasets, graph/index caches, model
weights or API credentials. Reproduction steps are given below; additional
runtime details are documented in [RUNNING.md](RUNNING.md).
Release-specific changes and validation scope are recorded in `CHANGELOG.md`
and `TESTING.md`. Upstream license notices are retained.

## Reproduction steps

### 1. Install the environment

Use Linux, Python 3.10 and an NVIDIA CUDA GPU with enough memory to load
NV-Embed-v2. Run all commands from the project root. The generation model is
Llama 3.3 70B Instruct, accessed through an OpenAI-compatible API.

```bash
cd /path/to/RGEC
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` records the original server's direct dependency versions.
The included `src/hipporag/` is imported directly; no `pip install .` step is
needed. vLLM is an import-time dependency even when generation uses the API.

### 2. Prepare the data, indexes and embedding model

The configurable entry point supports the following four datasets. The counts
below describe the historical experiment inputs recorded in `rgec/data_io.py`.

| Config key | Dataset | Sample JSON | Corpus JSON | Questions | Passages |
|---|---|---|---|---:|---:|
| `2wiki` | 2WikiMultiHopQA | `2wikimultihopqa.json` | `2wikimultihopqa_corpus.json` | 1,000 | 6,119 |
| `musique` | MuSiQue | `musique.json` | `musique_corpus.json` | 1,000 | 11,656 |
| `hotpotqa` | HotpotQA | `hotpotqa.json` | `hotpotqa_corpus.json` | 1,000 | 9,811 |
| `nq` | Natural Questions (NQ) | `nq_rear.json` | `nq_rear_corpus.json` | 1,000 | 9,633 |

Obtain the exact fixed 1,000-row evaluation splits and their corresponding
corpora and HippoRAG indexes for comparisons with the paper. This source-only
release does not contain those files, a download script or a split-generation
script. An arbitrary 1,000-question subset of an upstream dataset is a different
evaluation set. The configurable runner accepts other nonempty sample sets and
corpora of at least 200 passages; it does not enforce the historical counts.
Keep duplicate question rows, since evaluation preserves each original row.

The adapters in `rgec/data_io.py` expect:

- **Samples:** a JSON array of objects containing `question` and gold answers
  (`answer`, `gold_ans` or `reference`, as a string or a list of strings).
  `answer_aliases` is optional. Supporting passages must be represented by
  `supporting_facts` plus `context`, by `contexts` with `is_supporting`, or by
  `paragraphs` with the supporting annotations expected by the adapter.
- **Corpus:** a JSON array of passage strings or objects with `title` and
  `text`/`paragraph_text`. Objects are serialized as `title + "\n" + text`.
  Supporting passages must use the same text serialization as the corpus.
- **Offline index:** a matching OpenIE file, graph and three embedding stores
  for the exact corpus. Corpus passage texts must be unique. OpenIE passages
  and chunk-store contents must also have identical positional order.

For example, a prepared MuSiQue index must have this layout:

```text
/YOUR/INDEXES/musique/
  openie_results_ner_meta-llama_Llama-3.3-70B-Instruct.json
  meta-llama_Llama-3.3-70B-Instruct_nvidia_NV-Embed-v2/
    graph.pickle
    chunk_embeddings/vdb_chunk.parquet
    entity_embeddings/vdb_entity.parquet
    fact_embeddings/vdb_fact.parquet
```

The runner consumes prepared indexes and does not build them automatically.
If preparing new indexes, use `src.hipporag.HippoRAG.HippoRAG.index(docs)` as a
separate preprocessing step with matching OpenIE, graph and embedding settings;
`docs` must contain the serialized corpus passages. A new index is a changed
experimental input, not a substitute for the historical index in an exact
comparison. See [RUNNING.md](RUNNING.md) for the index contract.

Place NV-Embed-v2 in a local model directory, including its configuration,
tokenizer, custom Python code and weights. Loading uses local files and
`trust_remote_code=True`; the runner does not download model weights.

### 3. Configure paths and validate inputs

```bash
cp data_paths.example.json data_paths.json
```

Edit `data_paths.json` to point to your files. For a MuSiQue-only run, its entire
contents can be:

```json
{
  "musique": {
    "samples": "/YOUR/DATA/musique.json",
    "corpus": "/YOUR/DATA/musique_corpus.json",
    "offline": "/YOUR/INDEXES/musique"
  }
}
```

Keep all four entries from the template to run all four datasets sequentially.
Relative paths resolve against the configuration file's directory. The
`offline` path must point to the parent containing the OpenIE JSON and the
model-tagged index directory shown above.

```bash
python run_method.py --config data_paths.json --check
```

This checks source syntax, required files, hashes and corpus/OpenIE/chunk-store
alignment without API requests or model loading. It requires the prepared input
files and the dependencies used to read their Parquet stores.

### 4. Run RGEC

Set your API key, replace the model path, and run:

```bash
read -rsp 'Your API key: ' RGEC_API_KEY
export RGEC_API_KEY
printf '\n'

python -u run_method.py \
  --config data_paths.json \
  --embedding-model /YOUR/MODELS/NV-Embed-v2 \
  --gpu 0 \
  --api-url https://api.zhizengzeng.com/v1/chat/completions \
  --api-model llama-3.3-70b-instruct \
  --output runs/main
```

The API URL and model above are the entry point's defaults. Substitute your
OpenAI-compatible chat-completions endpoint and its model identifier if needed;
the returned model identity must match the requested identifier. A different
provider or backend may produce different outputs.

The run computes query inputs, retrieval and answers afresh. It retrieves 200
passages per question and supplies the top five original passages to the Reader,
with 400 output tokens, temperature 0 and seed 0. These are the main-experiment
settings. Compiler generation has a separate bounded output-token escalation.

### 5. Inspect results and resume

Results and logs are written under the selected `--output` directory:

| Path relative to `runs/main/` | Contents |
|---|---|
| `batch_manifest.json`, `inventory.json` | Run configuration and input identities |
| `embedding.log`, `<dataset>.log` | Embedding service and dataset execution logs |
| `<dataset>/progress.json` | Current stage and completed rows |
| `<dataset>/result.json` | Per-row predictions, QA EM/F1 and retrieval recall metrics |
| `<dataset>/completed.json` | Completion marker, result checksum and verification metadata |

For example, after a MuSiQue run, print its metrics with:

```bash
python - <<'PY'
import json
from pathlib import Path

result = json.loads(Path('runs/main/musique/result.json').read_text())
print(json.dumps({key: result[key] for key in ('qa_metrics', 'retrieval_metrics')}, indent=2))
PY
```

Rerun the identical launch command to resume from preserved checkpoints after
an interruption. If code, data, models or configuration change, choose a new
`--output` directory. Keep manifests intact so resume checks can verify the run.

For the included CPU source checks, run:

```bash
python test_source.py
```

These checks do not exercise CUDA, model loading or API inference. The release
has not been validated with a fresh full CUDA/API reproduction; see
[TESTING.md](TESTING.md). Exact historical results require matching experimental
inputs and deployment settings and are not guaranteed by this source package.
