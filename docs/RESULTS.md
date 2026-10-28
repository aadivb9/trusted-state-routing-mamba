# Results and unresolved scientific limits

| Question | Evidence | Interpretation |
| --- | --- | --- |
| Does routing help on the narrow synthetic schema? | Attacked two-choice 28/100 → 80/100; strict free generation 20/100 → 28/100 | An unfitted rule matches the cascade; the learned contribution is not isolated |
| Do two banks uniquely improve predictions? | Identical-decision filtering matches recorded outputs | No isolated predictive advantage |
| Did the new external study run? | [72 source examples and three controls](../experiments/17_external_validation/results_v2.json) | Strict-format competence gate failed; attacks were not generated |
| Is Mamba-2.8B feasible on T4? | Basic FP32/cache checks passed; [repeated run](../experiments/18_runtime_validation/timing_results.json) stopped at 166/240 paths | 128/512 blocks completed, 2,048-token block ran out of memory; no general memory-fit guarantee |
| Were annotations independently validated? | Different-model review reported 47/160 note-label and 5/32 candidate disagreements | No human adjudication; original labels unchanged. Raw abstract review data are not redistributed here |

The public saved-output checker recomputes strict and published-parser labels from the included external responses and checks the recorded timing failure. It cannot validate withheld source texts, reproduce the full private annotation audit, or establish fresh GPU inference. The full local package previously passed its consistency and isolated-extraction checks; that is a separate scope.

See the [paper](../paper/manuscript.md) for the full methods, conditional uncertainty, reference controls and limitations. The project supports a bounded implementation/negative-results study, not a validated general memory-poisoning defense.
