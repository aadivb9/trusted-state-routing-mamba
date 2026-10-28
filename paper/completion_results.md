<!-- COMPLETION_20261028 -->
### 5.10. Completed prospective screen and blinded model review

The October 28 T4 run used the frozen base Mamba-2.8B checkpoint, FP32, the fixed published-task prompts and unchanged competence criteria. It completed 72 new source-example generations and three instruction-only controls. The original runtime attempt failed before scoring a source example because the tokenizer joined trusted padding with the first untrusted word. A versioned adapter repair preserves the full prompt and its input token IDs; all 5,172 prepared records pass the exact AutoTokenizer serialization check. Both attempts and the repair are retained. No prompt, checkpoint, threshold or task was selected using these outcomes.

| Task | Examples | Published parser correct | Strict correct | Strict valid | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| SST-2 | 24 | 14/24 | 0/24 | 0/24 | Fail |
| SMS Spam | 24 | 0/24 | 0/24 | 0/24 | Fail |
| MRPC | 24 | 12/24 | 0/24 | 0/24 | Fail |

All three strict instruction-only gains are zero. The permissive published parser can recover labels from some paragraph answers; zero strict correctness therefore does not imply zero semantic answer content. The screen differs from the earlier development study in its source selection and fixed trailing-newline prompt, so the counts are not a controlled before/after comparison. The frozen strict-format gate failed, so the protocol stopped before attack-conditioned generation; before/after backbone hashes match. The 4,500 prepared attacks therefore provide no measured security efficacy result. This establishes a competence/format limitation of the tested checkpoint/prompt configuration. It does not settle whether a competent model with another preregistered configuration could benefit from routing.

A different-model blinded review used local Llama 3.2 3B at temperature zero with a pinned model digest. It received only the randomized V2 packet, one item per request, and completed 32 source items and 160 notes in 32 requests. Relative to the retained annotations it disagreed on 47/160 note labels and 5/32 supported candidates. These are disagreements, not an adjudicated error rate. Original labels remain unchanged. The abstract notes were generated/reviewed with Claude, while Llama had been used for an earlier separate synthetic pilot; this is bounded model/provider separation. No external human adjudication or returned human spot-check is available.

### 5.11. Repeated T4 timing and a memory failure

The follow-up planned three warmups per mode at each prefix length, then 20 timed repetitions per mode. It completed 36 warmups and 166/240 planned measured paths before CUDA out-of-memory failure during the 2,048-token block. The 128- and 512-token blocks each completed all 20 repetitions per mode; only six measured paths completed at 2,048 tokens. The full protocol failed, and its terminal weight hash is unavailable. Four mode orders are shuffled within paired repetitions with a fixed seed. Batch size is one, precision FP32 and generation length eight tokens. CUDA synchronization brackets full-path timing, including prefix processing, candidate-state creation/copy, speculative advancement, selection and generation. Loading, garbage collection and allocator-cache clearing are excluded. The decisions are forced, so this measures cache-path cost without learned-router cost or attack efficacy.

| Prefix tokens | Original median (s) | Forced accept median (s) | Forced reject median (s) | Filtering median (s) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.110 | 1.347 | 1.335 | 1.079 |
| 512 | 2.700 | 2.937 | 2.939 | 2.635 |

| Prefix tokens | Accept/original median paired time ratio | Reject/filter median paired time ratio |
| ---: | ---: | ---: |
| 128 | 1.212 | 1.234 |
| 512 | 1.086 | 1.105 |

The tables are post-failure descriptive summaries of the completely observed shorter blocks. All 80 paired output-token comparisons in those blocks agree. Pre-timing cache isolation/continuation checks passed. The planned complete-run uncertainty analysis was not reached; no complete-run interval or terminal weight check is claimed. Raw rows retain peak allocated/reserved memory, and the derived summary includes p95 values. The out-of-memory error reported 13.83 GiB allocated by PyTorch on a 14.56 GiB T4. Earlier short feasibility successes therefore do not guarantee repeated two-bank execution at 2,048 tokens. These observations condition on one workload, runtime and T4.
<!-- END_COMPLETION_20261028 -->
