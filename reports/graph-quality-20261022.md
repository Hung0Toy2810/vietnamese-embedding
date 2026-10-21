# Ten-page semantic graph and answer evaluation

## Result and scope

The final uninterrupted run passed all eight manually reviewed development checks:
six supported questions and two absent facts. All selected evidence quotes match
original source units and their expected subject scopes. See
[the answer review](graph-quality-20261022/answer-review.json) and
[the complete final outputs](graph-quality-20261022/answers-final.json).

This is a small, repeatedly used development set, not a held-out benchmark, legal
review, current-law certification or enterprise readiness claim. The input is a
partial ten-page excerpt of physical source PDF pages 70–79, from the
[135/VBHN-VPQH sample](https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/9/135-vbhn-vpqh.pdf).
Its SHA-256 is `83a918bbfe6cb68a1ae64a0f1bb159a3441bdef03c5372ede324f5ac88bcf697`.
Questions and expected source positions were fixed in
[cases.json](graph-quality-20261022/cases.json) before the final answer run.

There are only 79 children, fewer than the 100-candidate limit. Candidate retrieval
therefore covers this entire small excerpt. These results test grouping, reranking,
context handling and answers; they do not measure large-corpus encoder recall.
Context coverage is a diagnostic, not answer accuracy.

## Final answer review

| Check | Observed answer | Review |
|---|---|---|
| Broad theft overview | All four ranges, noncustodial alternative, thresholds, subthreshold conditions including antiques, OR alternatives and optional fine | Pass; all six expected source units are cited |
| Theft, 300 million | 7–15 years, expressed as a statutory range | Pass |
| Fraud, 600 million | 12–20 years or life | Pass |
| Breach of trust, 300 million | 5–12 years | Pass |
| Additional theft fine | Optional 5–50 million | Pass; no other offense's fine added |
| Snatching, 60 million | 3–10 years | Pass |
| Corporate tax rate in 2026 | Refusal, no invented rate or evidence | Pass |
| Severance benefit | Refusal, no invented formula or evidence | Pass |

Every answerable case read all its expected source units. Selected evidence atoms
were manually checked for the requested subject, range, threshold and modality.
The absent-fact cases returned no evidence and recorded their unsupported requests.
The one final run took 516 seconds locally; individual checks took about 49–116
seconds. These are descriptive timings with shared model/cache state, not a speed
benchmark or an isolated comparison of decoding modes.

## Graph checks

The evaluated version is `3cc79d8388c2449fbb90a9faf0bf0528`.
[graph-review.json](graph-quality-20261022/graph-review.json) records the artifact
hash, expected groups and mechanical checks.

- 79 source units, 17 subject anchors, 59 membership edges and one distant edge.
- All five required theft dependencies exist. None were available among the old 224 nearest-neighbor candidate pairs.
- All 57 selected substantive units have their expected offense scope.
- Every unit appears in exactly one indexed parent. Original text remains covered.
- Maximum child size is 186 encoder tokens including special tokens.
- 45 of 51 distant candidates received decisions; six omitted decisions created no edge.

Eight nodes have no semantic edge: five independent one-unit subjects and three
transitions/amendment annotations. They remain indexed. Connectivity is not forced.

The graph is imperfect. Several amendment notes have inaccurate relationship
labels, and the sole distant edge joins repeated amendment notes. It is a low-value
navigation link, not a dependency between the offenses. Some generated English
subject names are imprecise; retrieval uses original source openings. Extraction
can mix a continuing rule with a revision note or the next identifier. Identifier-only
queries, other domains and multi-turn factual quality are not validated by this set.

## What changed and why

The controlled graph comparison holds extraction, the 77M encoder embeddings and
the chunk plan constant. New production uploads still run the complete pipeline.

| Observed failure | Correction | Complexity review |
|---|---|---|
| Nearest-neighbor matching omitted all five required theft links | Read overlapping language windows, identify subjects, then assign units to fixed subjects | Reuses Qwen and source IDs; no new service or ontology |
| Similar fines in different offenses were treated as contradictory or exceptional | Preserve distinct subject scopes; only membership edges group parents | Connected components replace Louvain and its unused resolution parameter |
| Combined discovery/assignment made nearly every unit its own root | Keep subject discovery and membership separate; retain reviewed overlaps | Two explicit steps are necessary for the observed scope errors |
| Long generated endpoint quotes failed copying checks | Copy endpoint evidence in code from validated source IDs | Removes quote repair and wasted output tokens |
| Repeated wording produced false distant exceptions | Ask for explicit true/false decisions; omissions create no edge and are counted | Distant links never merge subject groups; their benefit remains unproven on this sample |
| Global IDs, echoed anchors and literal `"null"` caused avoidable failures | Use local aliases, validate/omit echoed anchors and normalize literal null | Unknown IDs and conflicting overlap scopes still fail |
| Writer mixed a correct theft fine with unrelated fines | Choose relevant subject groups before reading full parent passages | Uses the existing native JSON helper and same model; some related groups can still be overselected |
| Ranking changed the reading order of passages | Rank for selection, then restore original source order | A small correction, with no new model call |
| Split source identifiers caused false refusal | Reserve boundary-context space and allow one retry with preceding original units as numbered evidence | Reading context does not create graph edges or replace missing facts |
| Nested final-tool arguments and inline citation repair exhausted budgets | Separate tool-based source collection from native JSON answer writing; render citations per claim and copy evidence in code | Removes final-tool compatibility parsers; no second model/reviewer agent |
| Greedy answer generation repeated excessively | Use the model's sampled answer profile and bound generated fields | Existing extraction settings stay separate; source text is not truncated by answer-field bounds |
| An answerable result omitted all claims | Encode true/nonempty and false/empty alternatives in the provider schema | Enforces the existing validation rule before display |

The original baseline was rejected after two completed relation batches; it was
not a completed graph/answer benchmark. Intermediate runs include both successes
and failures. [trials.json](graph-quality-20261022/trials.json) summarizes them;
full raw trials are retained under the ignored local workspace. The completed
rejected navigation variant is recorded in
[rejected-navigation.json](graph-quality-20261022/rejected-navigation.json).
They must not be described as many complete eight-question comparisons.

The distant pass did not demonstrate answer benefit here. Its cost and noisy
annotation links remain limitations; no further machinery was added to make the
graph denser. The accepted stopping criterion is the document-answer task, not
visual connectivity. Further tuning should use fresh documents/questions rather
than repeatedly optimizing this same small set.

## Runtime and checks

The final run used local Qwen3.5:9B Q4_K_M through Ollama 0.40.6, with a 16,384-token
context. The [official Qwen3.5-9B guidance](https://huggingface.co/Qwen/Qwen3.5-9B#best-practices)
provides the non-thinking answer profile: temperature 0.7, top-p 0.8, top-k 20 and
presence penalty 1.5. The run fixes seed 42. Ingestion and short subject selection
retain their evaluated extraction profile. Settings changed alongside the answer
contract, so this is not an isolated sampling ablation.

Streams are consumed through completion without a read timeout or early token
cancellation. Schema and input budgets can reject incomplete/invalid output;
processing failure is distinguished from finding no supporting evidence.
Source/citation checks establish identity and exact text, not logical entailment.
Manual review remains necessary.

- 58 automated checks passed: [test output](graph-quality-20261022/automated-tests.txt).
- Pyright: zero errors/warnings: [type-check output](graph-quality-20261022/type-check.txt).
- Browser JavaScript passed `node --check`.
- Encoder checkpoint, tokenizer and protected historical metrics remain unchanged.

The exact evaluated graph was imported into the local web workspace without
regenerating it. Its 79 vectors were rebuilt with the matching encoder fingerprint;
vector count and a self-query were checked. HTTP checks confirmed the active
version, ten-page PDF checksum, graph, source preview and web assets. The workspace
is available at <http://127.0.0.1:8765>; see
[deployment.json](graph-quality-20261022/deployment.json) for artifact identity.
The earlier full-document ingestion was not resumed.

## Reproduction

From the repository root, with restored encoder artifacts and Ollama:

```bash
python reports/graph-quality-20261022/rebuild.py \
  legal_rag/data/graph-evaluation/baseline \
  legal_rag/data/graph-evaluation/revision-7
python reports/graph-quality-20261022/evaluate.py \
  legal_rag/data/graph-evaluation/revision-7 /tmp/answers.json --answers
```

The local baseline workspace supplies the frozen extraction, embeddings and plan.
Uploaded PDFs, vectors and model caches are intentionally excluded from Git.
Committed atoms/questions support inspection but are not a complete portable
workspace. Fresh ingestion may choose a different plan or model-generated graph.
The final evaluated artifact is retained for exact local inspection.
