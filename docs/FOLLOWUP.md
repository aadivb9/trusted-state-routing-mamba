# October 29 bounded follow-up

Review [larger sequence embeddings](../experiments/19_embedding_followup/README.md), [one-task SST-2](../experiments/20_sst2_followup/README.md), [the full computed summary](../remediation/followup_20261029/summary.json), and [the paper results](../paper/followup_results.md).

The three-encoder comparison uses sequence checkpoints, not pooled generic masked-LM outputs. Larger BGE improves AUROC in this fixed comparison; E5/span ranks slightly above MiniLM/span but rejects nothing at its selected threshold. Detection flags must be distinguished from removing attack text and preserving victim-task utility. All detector operating points remain exploratory.

The one-task study is separate from the old three-task failure and uses development/confirmation/reserved partitions. Its final status is `complete`. Both strict and permissive published parsers are retained. Same-mask filtering/two-bank equality is a control on implementation and attribution, not an architectural benefit.

The [author revision plan](../remediation/followup_20261029/AUTHOR_REVISION_PLAN.md) is for the next mentor meeting. The manuscript requires author revision before submission.

The public observation file deliberately omits raw third-party text and model responses. Its derived labels/counts can be checked, but response parsing, semantic payload grading, source-deduplication checks and fresh model inference require the complete local evidence package. Recorded inference hashes bind that package; this public snapshot is not an independent replication. The first runtime was lost: 534 recovered rows and the later missing rows count once, while 24 exact overlap replays are controls. The interrupted segment has no terminal weight hash.

| Arm | Strict clean correct | Strict attacked correct |
| --- | ---: | ---: |
| undefended | 44/50 | 118/250 |
| unfitted_rule_filtering | 44/50 | 115/250 |
| learned_span_filtering | 44/50 | 118/250 |
| learned_span_two_bank | 44/50 | 118/250 |
