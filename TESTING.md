# RGEC source checks

Checks run without API calls, GPU calls, model weights or graph deserialization:

- All included Python modules parse successfully.
- The entry-point help runs without datasets or GPU dependencies being loaded.
- External paths are resolved correctly and input file identities are recorded.
- Empty/partial evidence routes preserve all ranked passages without additional
  gap retrieval, route reconstruction or model calls.
- The original semantic candidate order is preserved before recognition filtering.
- README files contain no description of the removed algorithmic module.
- Archive contents are checked for old method identifiers, Chinese characters,
  personal names/paths, institution labels and known credential patterns.
- Runtime source imports are statically checked for missing local modules.

`python test_source.py` runs seven CPU tests. No clean Linux CUDA installation,
actual NV-Embed-v2 loading, Llama API generation or full experiment reproduction
was performed while preparing this release. Test fixtures do not represent
measured experimental results. See CHANGELOG.md for the source-version scope.
