# Mentor review guide

Start with the [paper](../paper/manuscript.pdf) and [result summary](RESULTS.md). This public copy supplies the implementation and selected records; omitted licensed/raw/private inputs are described in [data availability](../data/README.md).

1. [State controller](../src/memory/state_controller.py): inspect `install_chunk_continuation`, `make_state_access`, `gpu_check`, and `answer_once`. The backbone is frozen; an external router controls the two cache banks.
2. [Conservative router](../src/routing/guarded_router.py) and [original symbolic router](../src/routing/context_router.py): check observable inputs, unsupported attributes, quoted attacks, abstention and utility costs.
3. [Embedding comparison](../src/routing/embedding_router.py) and [calibration](../src/validation/calibration.py): distinguish held-out-domain detection from task utility and inspect the per-domain false-positive issue.
4. [External runner](../experiments/17_external_validation/run_v2.py), [frozen gate](../experiments/17_external_validation/protocol.json), [raw responses](../experiments/17_external_validation/results_v2.json): the failed competence screen supplies no attack-efficacy estimate.
5. [Timing driver](../experiments/18_runtime_validation/run_timing.py), [plan](../experiments/18_runtime_validation/timing_protocol.json), [failed output](../experiments/18_runtime_validation/timing_results.json): the full protocol failed at 2,048 tokens. No missing measurements or terminal weight check are presented as completed.
6. [Blinding exporter](../src/validation/blinded_review_v2.py) and [model-review runner](../remediation/completion_20261027/review_annotations.py): packet contents, mappings and request logs are excluded from this public copy. The reported different-model disagreements remain unadjudicated.

The earlier 100% published-trigger result uses an easy two-choice adaptation. Empty-context and evidence-removal controls prevent interpreting it as successful defense. No full official RoBench reproduction or superiority over a published defense is claimed.

`archive/sealed_workspace/` is retained as a source import layout in this export. It contains selected first-party source, not the complete sealed archive. Nine exported controller copies replace the embedded upstream extractor with its pinned lazy download. The adaptation is recorded explicitly rather than changing any historical result or claiming a fresh replication.

The next bounded round is in [the October 29 follow-up guide](FOLLOWUP.md). Review that before using the historical competence failure as the current one-task result.
