import json
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='Write bounded public follow-up documentation');parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args();P=args.destination.resolve()
def put(n,s):p=P/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def append_once(n,text):
 p=P/n;s=p.read_text()
 if text not in s:p.write_text(s+text)
s=json.loads((P/'remediation/followup_20261029/summary.json').read_text());e=s['external']
put('experiments/19_embedding_followup/README.md','''# Larger sequence-embedding comparison

MiniLM (22.7M parameters), BGE-base (109.5M) and E5-base (109.5M) were compared with document/span heads. All are embedding-trained sequence checkpoints. BGE uses CLS pooling; E5 uses mean pooling with its classification query prefix. Revisions and frozen methods are in [the protocol](protocol.json) and [encoder code](../../src/routing/embedding_followup.py).

BGE/span was selected using known-domain calibration before the new summarization-domain features were computed. Final AUROC is 0.6373; 55/600 poisoned documents and 5/100 clean documents are flagged, but payload text is removed in only 27/600 poisoned documents. The other five policies reject nothing at their selected thresholds. All six fail the unchanged confidence-support requirement. This is detection on BIPIA-derived data, not official BIPIA victim-task success scoring.

[Fitted linear heads](frozen_heads.json), [the original detector score record](results.json), and [frozen-head calibration reconstruction](calibration_replay.json) are included. [Derived scores](../../provenance/followup_observations.json) contain labels, source grouping, scores and removal counts without raw third-party text. [The recomputed summary](../../remediation/followup_20261029/summary.json) records conditional source-bootstrap intervals. Public checks verify these included derived observations; they do not re-encode sources or independently establish text-level payload overlaps. The full local mentor package retains original corpora, construction offsets, responses and complete auditors. This public export cannot run full `tools/followup_summary.py` without those inputs.
''')
put('experiments/20_sst2_followup/README.md',f'''# One-task SST-2 study

A separate bounded development study chose four-shot prompting at 34/40 strict correct; the fixed format then scored 87/100 strict correct and 100/100 valid on fresh confirmation. The prior three-task failure remains unchanged. See [the screen protocol](protocol.json), [final protocol](final_protocol.json), and [derived observations](../../provenance/followup_observations.json).

Reserved evaluation ended with status `{e['status']}` and {e['rows']} saved generations. [The summary](../../remediation/followup_20261029/summary.json) includes clean utility, attacked utility, unfitted filtering and matched learned-filter/two-bank controls. Only 50 target sources, five constructors, one injected task and one selected prompt were evaluated. A short injected source has a punctuation-only variant in prior data; no target or prompt-role overlap was found under the recorded checks. The published spam instruction asks for yes/no while its parser expects spam/not-spam; compatible yes/no scoring is explicitly post hoc.

Raw source text, masks, model responses and reversible generated token IDs are omitted here. Derived prediction labels and generated-output hashes support bounded public count/equivalence checks, not independent response parsing or fresh inference. Exact reproduction also needs complete licensed inputs and original sealed controller bytes; the public controller adapts an embedded third-party extractor to a pinned lazy download, so original protocol hashes intentionally do not silently accept its changed bytes. The run recovered 534 generations after a Colab VM replacement, then continued with 24 exact overlap replays and matching continuation weight hashes. The interrupted segment has no terminal weight hash; overlap does not fill that gap. Earlier full timing OOM and unverified native instruction-checkpoint conversion remain limitations.
''')
p=P/'README.md';t=p.read_text();t=t.replace('The paper remains a research draft. External attack efficacy, human-adjudicated annotation accuracy, native instruction-checkpoint conversion equivalence and full official RoBench/BIPIA task-utility evaluations remain unestablished.',f'''The [October 29 follow-up](docs/FOLLOWUP.md) adds two larger sequence embedding checkpoints and a separate SST-2-only study. BGE improves ranking modestly but removes payload text in only 27/600 poisoned BIPIA-derived documents; E5 rejects nothing at its selected operating points. The selected four-shot Mamba format passes fresh confirmation at 87/100. Reserved attack execution ended `{e['status']}`; see the linked record for the full four-arm comparison.

The paper remains a research draft. A general poisoning defense, a predictive benefit unique to two banks, human-adjudicated annotation accuracy, native instruction-checkpoint conversion equivalence and full official RoBench/BIPIA task-utility evaluations remain unestablished.''');p.write_text(t)
put('docs/FOLLOWUP.md',f'''# October 29 bounded follow-up

Review [larger sequence embeddings](../experiments/19_embedding_followup/README.md), [one-task SST-2](../experiments/20_sst2_followup/README.md), [the full computed summary](../remediation/followup_20261029/summary.json), and [the paper results](../paper/followup_results.md).

The three-encoder comparison uses sequence checkpoints, not pooled generic masked-LM outputs. Larger BGE improves AUROC in this fixed comparison; E5/span ranks slightly above MiniLM/span but rejects nothing at its selected threshold. Detection flags must be distinguished from removing attack text and preserving victim-task utility. All detector operating points remain exploratory.

The one-task study is separate from the old three-task failure and uses development/confirmation/reserved partitions. Its final status is `{e['status']}`. Both strict and permissive published parsers are retained. Same-mask filtering/two-bank equality is a control on implementation and attribution, not an architectural benefit.

The [author revision plan](../remediation/followup_20261029/AUTHOR_REVISION_PLAN.md) is for the next mentor meeting. The manuscript requires author revision before submission.

The public observation file deliberately omits raw third-party text and model responses. Its derived labels/counts can be checked, but response parsing, semantic payload grading, source-deduplication checks and fresh model inference require the complete local evidence package. Recorded inference hashes bind that package; this public snapshot is not an independent replication. The first runtime was lost: 534 recovered rows and the later missing rows count once, while 24 exact overlap replays are controls. The interrupted segment has no terminal weight hash.
''')
append_once('docs/MENTOR_REVIEW.md','\nThe next bounded round is in [the October 29 follow-up guide](FOLLOWUP.md). Review that before using the historical competence failure as the current one-task result.\n')
append_once('docs/RESULTS.md','\nSee [the October 29 follow-up](FOLLOWUP.md) for larger sequence encoders and the separate SST-2 competence/attack comparison. Earlier protocols remain historical outcomes.\n')
append_once('data/README.md','\nOctober 29 adds BIPIA summarization/XSum-derived contexts and new SST-2/SMS selections. Raw corpora, model responses, masks and reversible generated IDs are omitted; public derived observations support count checks only. Dataset access does not imply unrestricted redistribution.\n')
append_once('NOTICE.md','\nOctober 29 sequence encoders: BAAI/bge-base-en-v1.5 and intfloat/e5-base-v2 are referenced at pinned revisions; weights are not distributed. BIPIA summarization uses XSum/BBC-derived text locally. No XSum or new SST-2/SMS source text is redistributed in this update.\n')
