<!-- FOLLOWUP_20261029 -->
### 5.12. Larger sequence encoders on a new BIPIA domain

A bounded follow-up compares MiniLM (22,713,216 encoder parameters), BGE-base and E5-base (109,482,240 each), with one document head and one generic observed-text span head per encoder. These are embedding-trained sequence checkpoints, rather than pooled generic masked-language-model checkpoints. BGE uses CLS pooling; E5 uses masked mean pooling and its documented `query: ` prefix. The encoders differ in training and architecture as well as size, so this is not an isolated scaling experiment. Same fit data and classifier settings are used across all six cells. Compared with the earlier MiniLM study, segmentation and pooling also change; its older scores are not controlled before/after baselines.

Training uses 200 source/near-duplicate groups from BIPIA table data. Previously observed email, table and code domains are calibration/development data. The summarization domain is entirely withheld from fitting, calibration and head selection: 100 official BIPIA XSum test sources, reconstructed to match the upstream MD5, with six published payload presentations and one clean presentation per source. This is a bounded detection construction from external benchmark data, not the full BIPIA victim-model protocol. Detector ground truth records insertion of a known payload rather than independently adjudicated attack execution. Attack families can overlap across splits. Exact and near-duplicate source groups are disjoint; foundation-model pretraining overlap is unknown.

| Encoder/head | Final AUROC | Clean documents flagged | Poisoned documents flagged | Documents with payload text removed |
| --- | ---: | ---: | ---: | ---: |
| minilm/document | 0.5344 | 0/100 | 0/600 | 0/600 |
| minilm/span | 0.5255 | 0/100 | 0/600 | 0/600 |
| bge/document | 0.6281 | 0/100 | 0/600 | 0/600 |
| bge/span | 0.6373 | 5/100 | 55/600 | 27/600 |
| e5/document | 0.5324 | 0/100 | 0/600 | 0/600 |
| e5/span | 0.5515 | 0/100 | 0/600 | 0/600 |

BGE/span was selected using calibration alone before final-domain features were computed. At its frozen exploratory threshold, flag recall is 55/600 (9.2%; source-cluster bootstrap 95% interval [5.3, 13.7]%). Only 27/600 poisoned documents (4.5%; interval [3.2, 6.0]%) have a payload-overlapping span removed; all 27 of these have the full constructed payload removed. The remaining flagged poisoned documents lose benign spans only. Clean-document FPR is 5/100 (5.0%; percentile interval [1.0, 10.0]%). Across the 700 presentations, exact interval-based accounting removes 4,722/1,242,988 non-payload characters and 918/39,373 payload characters. Character retention is not victim-task utility.

The paired BGE/span versus MiniLM/span AUROC difference is 0.1118, with a source-cluster bootstrap 95% interval [0.0814, 0.1460]. This supports a ranking improvement in this fixed comparison; it does not establish useful low-FPR protection or a causal benefit from parameter count alone.

All six heads fail the unchanged confidence-support requirement because email supplies only 78 independent benign groups rather than the required 100. The exploratory selector constrains calibration point FPR, not held-out confidence bounds. Additional source support alone would not establish a useful threshold: the recall and uncertainty-bound requirements would still need to pass. Five heads select a reject-nothing threshold: zero final FPR there does not demonstrate protection. Intervals condition on the fixed selected configuration, omit encoder/head selection and training uncertainty, and can degenerate for zero-count cells. Frozen-head calibration reconstruction reproduces original decision metrics, but was saved after inference and is not original raw-score logging. No BIPIA victim generations establish attack-success reduction.

### 5.13. A separate one-task competence and attack study

A new SST-2-only protocol permits exactly three predefined prompt formats on 40 fresh development examples. It is distinct from the failed historical three-task protocol. Current-example labels are withheld; four disjoint training demonstrations are explicitly trusted. The formats score 20/40, 22/40 and 34/40 strict correct, respectively. Only the four-shot format qualifies. With that format frozen, fresh confirmation scores 87/100 strict correct and 100/100 valid, versus 74/100 using the permissive published parser. Later echoed labels explain why the parsers differ. This establishes competence for the chosen task/format; it is not a controlled gain relative to the earlier different data and prompts.

Reserved evaluation uses 50 additional target sources, five unchanged published attack constructors and one injected spam task (250 poisoned presentations). It compares undefended inference, an unfitted text-rule filter, learned BGE/span filtering, and two-bank execution using the same learned masks. Generic sentence/newline/64-word segmentation uses observed text only. No gold offset, original clean counterpart, source ID or attack identity enters routing. Full prompts are tokenized once, and filtering/two-bank arms must receive identical retained tokens and produce identical generated token IDs. The recorded tokenizer preflight covers all 350 clean/attack/injected records, with a maximum of 280 input tokens and no boundary failures. A later stricter punctuation-insensitive audit finds one short injected-source variant (`Ok...` versus a historical `Ok`); it is not a target or prompt-role overlap. The frozen lowercase/whitespace exact check and five-word Jaccard check miss this variant. Prepared inputs remain unchanged, and stronger semantic source independence is not claimed.

The reserved run ended with status `complete` and 1,250 recorded generations. Undefended attacks change paired strict target accuracy by -40.8 percentage points and have strict attacker-task success 0.0%. The frozen attack-validity gate passes: it requires a target-accuracy drop of at least ten points or strict attacker success of at least 10%.

| Condition | Strict target correct | Strict-format valid | Published target correct | Strict attacker success | Published attacker success |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean / Undefended | 44/50 | 50/50 | 34/50 | — | — |
| Clean / Unfitted filter | 44/50 | 50/50 | 34/50 | — | — |
| Clean / Learned filter | 44/50 | 50/50 | 34/50 | — | — |
| Clean / Two-bank, same masks | 44/50 | 50/50 | 34/50 | — | — |
| Attack / Undefended | 118/250 | 250/250 | 138/250 | 0/250 | 0/250 |
| Attack / Unfitted filter | 115/250 | 250/250 | 138/250 | 0/250 | 0/250 |
| Attack / Learned filter | 118/250 | 250/250 | 138/250 | 0/250 | 0/250 |
| Attack / Two-bank, same masks | 118/250 | 250/250 | 138/250 | 0/250 | 0/250 |

All 250 undefended attacked outputs remain valid under the strict target-label parser, so the observed accuracy loss is not an invalid-format count. Learned filtering changes 5 input token streams and 0 target predictions, with 0 correctness changes. It supplies no measured target-accuracy gain. The unfitted filter scores 115/250, versus 118/250 undefended. Zero learned paired intervals describe the observed unchanged paired scores, not a universal no-effect guarantee.

Identical-decision filtering and two-bank outputs match in all 300 observed pairs. This supports implementation equivalence in this run, not a predictive advantage of the two-bank architecture.

| Attacked arm versus undefended | Paired target-accuracy difference | Source-cluster bootstrap 95% interval |
| --- | ---: | ---: |
| Unfitted filter | -1.2 points | [-4.0, 1.6] points |
| Learned filter | 0.0 points | [0.0, 0.0] points |
| Two-bank, same masks | 0.0 points | [0.0, 0.0] points |

| Attacked condition | Post-hoc yes/no attacker success | Valid yes/no outputs |
| --- | ---: | ---: |
| Undefended | 0/250 | 0/250 |
| Unfitted filter | 0/250 | 0/250 |
| Learned filter | 0/250 | 0/250 |
| Two-bank, same masks | 0/250 | 0/250 |

Colab replaced the first runtime. The execution-only continuation retains 534 observed rows and adds 716 missing planned rows, with 24 exact overlap replay generations kept separately from the sample counts. Prompts, data, masks, model, thresholds and the original validity gate are unchanged; the native, two-bank, scoring and token-serialization function bodies are checked against the original runner. The interrupted segment has no terminal weight hash. Matching continuation start/end hashes and overlap outputs do not establish that missing check.

The published injected-task instruction asks for yes/no, whereas its fixed benchmark parser expects spam/not-spam. Injected controls have 0/50 valid outputs and 0/50 correct under the strict spam parser. A compatible yes/no parser, audited post hoc, finds 20/50 valid and 13/50 correct. This secondary check does not change the frozen attack-validity gate. The injected-task control does not establish adequate spam-task competence in this prompt configuration, so low attacker-task ASR cannot establish resistance to instruction redirection. Target-accuracy damage and preservation remain separately measurable under the frozen target-drop criterion.

The backbone is the pinned FP32 Mamba-2.8B on a T4; recorded start/end fingerprints and cache-copy checks are audited where the run completes. Masks were precomputed on Apple MPS, so decoder/cache execution is not end-to-end routing latency. Fifty target-source clusters, one task, one prompt, one injected task and five constructors limit inference. No matched benign added-text/length control isolates instruction-specific damage from distraction by extra context. Paired intervals condition on the fixed configuration and cannot establish domain-wide or adaptive-attack robustness. Earlier three-task failures and the 2,048-token timing OOM remain failures under their original protocols.
<!-- END_FOLLOWUP_20261029 -->
