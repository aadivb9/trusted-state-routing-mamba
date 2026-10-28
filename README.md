# Trusted-state routing in frozen Mamba

Research code and a manuscript draft investigating whether a trusted/candidate pair of recurrent-state banks can reduce memory-poisoning damage without discarding useful context.

**Current conclusion:** the measured benefit is narrow. An unfitted rule matches the trained cascade's decisions on the observed synthetic set, and identical-decision filtering matches recorded controller outputs. A general poisoning defense or a predictive benefit unique to two banks has not been established.

## Review the work

- [Manuscript PDF](paper/manuscript.pdf) · [editable manuscript](paper/manuscript.md)
- [Mentor code-review guide](docs/MENTOR_REVIEW.md)
- [Results and limitations](docs/RESULTS.md)
- [Implementation](src/README.md)
- [Data sources and public-release scope](data/README.md)
- [Third-party attribution and reuse terms](NOTICE.md)

The strongest synthetic attacked two-choice result is 28/100 → 80/100. Strict free generation is 20/100 → 28/100, with paired interval [0,17] percentage points. These metrics are different and do not isolate a learned-router or two-bank contribution.

The October 28 T4 competence screen completed 72 source examples. The fixed strict-format gate failed on SST-2, SMS Spam and MRPC, stopping attack-conditioned evaluation. The permissive published parser scored 14/24, 0/24 and 12/24; zero strict correctness does not mean zero semantic answer content. A separate repeated timing protocol failed from out-of-memory at 2,048 tokens after 166/240 measurements. Completed 128/512-token blocks are descriptive, not a passed truncated protocol.

## Run the public checks

```sh
python3 tools/verify_public.py
python3 tools/run_public_checks.py
```

These standard-library commands verify this export and re-score included saved outputs, then run the data-independent implementation tests. They perform no new model inference. The internal `tools/verify.py` and historical full-suite commands require the complete sealed evidence workspace and are deliberately not substituted by the public checks.

## Public export versus sealed evidence

This repository contains first-party Python implementations and experiment runners, selected result/protocol records, and the current paper. Raw benchmark corpora, abstract packets, private annotation mappings/seeds, browser sessions, local connector settings, generated notebook payloads and historical ZIPs are excluded. [Source provenance](provenance/export_origins.json) records the original/export hashes and explicit code adaptations.

The embedded third-party HiSPA extractor was replaced only in this public copy with a lazy HTTPS download at a pinned commit and verified SHA-256. Importing the controller does not download it. Historical runners retain their original protocol hashes and fail closed if the required exact sealed inputs are unavailable; this export is not a byte-identical, fully self-contained reproduction of all historical experiments. The original local archive and verified mentor package were left unchanged.

The paper remains a research draft. External attack efficacy, human-adjudicated annotation accuracy, native instruction-checkpoint conversion equivalence and full official RoBench/BIPIA task-utility evaluations remain unestablished.
