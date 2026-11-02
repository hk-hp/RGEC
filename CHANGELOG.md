# RGEC source release

## rgec-source-v1

- Standardized the method name, module names, class names, environment variables
  and run-local artifact names to RGEC, matching the paper terminology.
- Organized the entry implementation as `rgec/method.py`, shared evidence logic
  as `rgec/evidence_compiler.py` and dataset adapters as `rgec/data_io.py`.
- Excluded historical deployment utilities, old experiment reports and unused
  source-comparison files from this main-method distribution.
- Removed an unreachable base-class search implementation that was always
  overridden by the submitted method. The active recognition-input preparation
  retains semantic Top30 order and the same LLM fact filtering.
- Kept the requested zero-round algorithmic recovery setting. Missing-slot
  search, route reconstruction and the semantic recovery prompt are absent.
  Network/truncation/format retries remain bounded and separate.
- Kept configurable model/API paths and consistent terminal-JSON parsing.
- Preserved English display text, anonymous paths and neutral index namespaces.
  Upstream third-party attribution is retained.

Renaming changes code hashes, module names, environment-variable names and
checkpoint filenames. Use a new output directory; do not adopt old manifests
or manually rename earlier checkpoints to bypass identity checks.

Historical results are not rewritten by these changes. This source variant was
not verified by a fresh full CUDA/API experiment. CPU tests and source checks
do not establish that every historical request used this edited source version.
