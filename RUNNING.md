# Relation-Guided Evidence Compilation (RGEC)

Main-method source for **Relation-Guided Evidence Compilation for GraphRAG under
a Fixed Passage Budget**. This package provides the method implementation,
original prompts/evaluators, HippoRAG source dependencies and a configurable
entry point. It does not include datasets, graph/index caches, model weights,
experimental results or API credentials.

## Method-to-code map

| Paper component | Main implementation |
|---|---|
| 3.2 Query requirements and evidence representation | `rgec/query_inputs.py`; `rgec/evidence_compiler.py::_query_slots` |
| 3.3 Query-conditioned candidate acquisition | `rgec/method.py::_prepare_recognition_candidates`; `rgec/evidence_compiler.py::_query_conditioned_candidate_acquisition` and `run_ppr` |
| 3.4 Source-grounded evidence compilation | `rgec/evidence_compiler.py::_triple_alignment_scores`, `_compile_evidence`, `_valid_connected_chain` |
| 3.5 Budgeted context assembly | `rgec/evidence_compiler.py::retrieve` |
| Dataset adapters and evaluation | `rgec/data_io.py`; `rgec/experiment_runtime.py`; `src/hipporag/evaluation/` |

`RGEC` in `rgec/method.py` is the submitted method class. `EvidenceCompiler` in
`rgec/evidence_compiler.py` implements its shared candidate acquisition,
assignment validation and context assembly. Upstream HippoRAG names and license
notices are retained rather than relabeled as original RGEC code.

## Environment

Use Linux, Python 3.10 and an NVIDIA CUDA GPU with enough memory for NV-Embed-v2.
Llama 3.3 70B is accessed through an OpenAI-compatible API, so local 70B weights
are not required. Datasets run sequentially with one Embedding service.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Run the entry point from this directory, not `pip install .`. The included
`src/hipporag` source is used directly. vLLM remains an import-time dependency of
the upstream framework even though this main-method entry uses the API.

## Prepare external inputs

Supply sample JSON, corpus JSON and a matching HippoRAG index for each dataset.
Index construction is provided by
`src.hipporag.HippoRAG.HippoRAG.index(docs)` as a separate preprocessing step.
The RGEC subclass intentionally consumes prepared indexes; it does not rebuild
them implicitly. Build indexes from the exact intended corpus using matching
OpenIE, graph and NV-Embed-v2 settings. Do not substitute unrelated caches.

Each index folder must contain:

```text
openie_results_ner_meta-llama_Llama-3.3-70B-Instruct.json
meta-llama_Llama-3.3-70B-Instruct_nvidia_NV-Embed-v2/
  graph.pickle
  chunk_embeddings/vdb_chunk.parquet
  entity_embeddings/vdb_entity.parquet
  fact_embeddings/vdb_fact.parquet
```

The tag is this release's neutral index namespace, not a required absolute
model location. The actual model path is configured at launch. Use only trusted
graph pickles. Data/model use remains subject to the respective terms.

Fill `data_paths.example.json` with paths on your own machine. Relative paths
resolve against that JSON's directory. Remove datasets you do not want to run.
Use the fixed 1,000-row splits for paper comparisons; this source-only package
does not ship those splits. Duplicate question rows remain separate evaluation
rows. Different corpus/index/model versions constitute a changed experiment.

## Run

```bash
read -rsp 'Your API key: ' RGEC_API_KEY; export RGEC_API_KEY; printf '\n'
python run_method.py --config your_paths.json --check
python -u run_method.py --config your_paths.json \
  --embedding-model /YOUR/MODELS/NV-Embed-v2 --gpu 0
```

The model directory must include configuration, tokenizer, trusted custom Python
code and weights. Model loading uses local files and `trust_remote_code=True`.
No weights are downloaded automatically. Use your own key; do not publish it.
Override the API URL/model with `--api-url`/`--api-model` if needed. Returned
model identity must match the requested identifier. Changed providers/backends
are changed deployments, not a guarantee of identical outputs.

This entry freshly computes query inputs, retrieval and QA. It does not replay
historical rankings or answers. Reader uses five passages, 400 output tokens,
temperature 0 and seed 0. Compiler generation retains bounded output escalation
up to 8,192; this is not an 8,192-token Reader. API/format failures preserve
checkpoints and pause rather than silently exclude questions from scoring.

Logs/results are under `runs/main`. Restart the same command to resume. Changed
code/data/model/configuration requires a new `--output` directory; never delete
the manifest to bypass identity checks. No author's server access is required.

## Source validation

```bash
python test_source.py
```

These checks exercise source syntax, configuration, entry-point help and isolated
CPU method behavior without API/GPU/model calls. They are not an end-to-end
CUDA/API reproduction. No clean Linux CUDA installation or fresh full inference
was run on the packaging machine. See `TESTING.md` and `CHANGELOG.md` for scope
and version-specific changes. Exact historical numbers are not guaranteed by
a code-only submission.

All source comments and user-facing messages are English. Provider-error
matching strings use Unicode escapes to preserve classification semantics.
