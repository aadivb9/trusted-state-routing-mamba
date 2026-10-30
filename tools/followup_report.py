"""Generate the current follow-up table and narrative from recomputed evidence."""
from pathlib import Path


def percent(value):
    return f'{100 * value:.1f}%'


def interval(values):
    return f'[{100 * values[0]:.1f}, {100 * values[1]:.1f}]'


def arm_label(key):
    labels={'undefended':'Undefended','unfitted_rule_filtering':'Unfitted filter','learned_span_filtering':'Learned filter','learned_span_two_bank':'Two-bank, same masks'}
    if '/' in key:
        role,arm=key.split('/',1)
        return role.capitalize()+' / '+labels[arm]
    return labels[key]


def paper_block(s):
    d=s['detector'];screen=s['screen'];e=s['external'];b=d['cells']['bge/span']
    lines=['<!-- FOLLOWUP_20261029 -->','### 5.12. Larger sequence encoders on a new BIPIA domain','',
    'A bounded follow-up compares MiniLM (22,713,216 encoder parameters), BGE-base and E5-base (109,482,240 each), with one document head and one generic observed-text span head per encoder. These are embedding-trained sequence checkpoints, rather than pooled generic masked-language-model checkpoints. BGE uses CLS pooling; E5 uses masked mean pooling and its documented `query: ` prefix. The encoders differ in training and architecture as well as size, so this is not an isolated scaling experiment. Same fit data and classifier settings are used across all six cells. Compared with the earlier MiniLM study, segmentation and pooling also change; its older scores are not controlled before/after baselines.','',
    'Training uses 200 source/near-duplicate groups from BIPIA table data. Previously observed email, table and code domains are calibration/development data. The summarization domain is entirely withheld from fitting, calibration and head selection: 100 official BIPIA XSum test sources, reconstructed to match the upstream MD5, with six published payload presentations and one clean presentation per source. This is a bounded detection construction from external benchmark data, not the full BIPIA victim-model protocol. Detector ground truth records insertion of a known payload rather than independently adjudicated attack execution. Attack families can overlap across splits. Exact and near-duplicate source groups are disjoint; foundation-model pretraining overlap is unknown.','',
    '| Encoder/head | Final AUROC | Clean documents flagged | Poisoned documents flagged | Documents with payload text removed |',
    '| --- | ---: | ---: | ---: | ---: |']
    for key,c in d['cells'].items():
        lines.append(f'| {key} | {c["auroc"]:.4f} | {c["clean_flagged"]}/100 | {c["attack_flagged"]}/600 | {c["payload_hit"]}/600 |')
    lines += ['',f'BGE/span was selected using calibration alone before final-domain features were computed. At its frozen exploratory threshold, flag recall is {b["attack_flagged"]}/600 ({percent(b["attack_flagged"]/600)}; source-cluster bootstrap 95% interval {interval(b["recall_ci95"])}%). Only {b["payload_hit"]}/600 poisoned documents ({percent(b["payload_hit"]/600)}; interval {interval(b["payload_hit_ci95"])}%) have a payload-overlapping span removed; all {b["full_payload_removed"]} of these have the full constructed payload removed. The remaining flagged poisoned documents lose benign spans only. Clean-document FPR is {b["clean_flagged"]}/100 ({percent(b["clean_flagged"]/100)}; percentile interval {interval(b["fpr_ci95"])}%). Across the 700 presentations, exact interval-based accounting removes {b["exact_benign_characters_removed"]:,}/{b["exact_benign_characters"]:,} non-payload characters and {b["payload_characters_removed"]:,}/{b["payload_characters"]:,} payload characters. Character retention is not victim-task utility.', '',
    f'The paired BGE/span versus MiniLM/span AUROC difference is {d["auroc_comparison"]["auroc_difference"]:.4f}, with a source-cluster bootstrap 95% interval [{d["auroc_comparison"]["paired_source_bootstrap_ci95"][0]:.4f}, {d["auroc_comparison"]["paired_source_bootstrap_ci95"][1]:.4f}]. This supports a ranking improvement in this fixed comparison; it does not establish useful low-FPR protection or a causal benefit from parameter count alone.', '',
    'All six heads fail the unchanged confidence-support requirement because email supplies only 78 independent benign groups rather than the required 100. The exploratory selector constrains calibration point FPR, not held-out confidence bounds. Additional source support alone would not establish a useful threshold: the recall and uncertainty-bound requirements would still need to pass. Five heads select a reject-nothing threshold: zero final FPR there does not demonstrate protection. Intervals condition on the fixed selected configuration, omit encoder/head selection and training uncertainty, and can degenerate for zero-count cells. Frozen-head calibration reconstruction reproduces original decision metrics, but was saved after inference and is not original raw-score logging. No BIPIA victim generations establish attack-success reduction.','',
    '### 5.13. A separate one-task competence and attack study','',
    'A new SST-2-only protocol permits exactly three predefined prompt formats on 40 fresh development examples. It is distinct from the failed historical three-task protocol. Current-example labels are withheld; four disjoint training demonstrations are explicitly trusted. The formats score 20/40, 22/40 and 34/40 strict correct, respectively. Only the four-shot format qualifies. With that format frozen, fresh confirmation scores 87/100 strict correct and 100/100 valid, versus 74/100 using the permissive published parser. Later echoed labels explain why the parsers differ. This establishes competence for the chosen task/format; it is not a controlled gain relative to the earlier different data and prompts.','',
    'Reserved evaluation uses 50 additional target sources, five unchanged published attack constructors and one injected spam task (250 poisoned presentations). It compares undefended inference, an unfitted text-rule filter, learned BGE/span filtering, and two-bank execution using the same learned masks. Generic sentence/newline/64-word segmentation uses observed text only. No gold offset, original clean counterpart, source ID or attack identity enters routing. Full prompts are tokenized once, and filtering/two-bank arms must receive identical retained tokens and produce identical generated token IDs. The recorded tokenizer preflight covers all 350 clean/attack/injected records, with a maximum of 280 input tokens and no boundary failures. A later stricter punctuation-insensitive audit finds one short injected-source variant (`Ok...` versus a historical `Ok`); it is not a target or prompt-role overlap. The frozen lowercase/whitespace exact check and five-word Jaccard check miss this variant. Prepared inputs remain unchanged, and stronger semantic source independence is not claimed.','']
    if e['status'] in ['complete','stopped_attack_validity_screen']:
        g=e['attack_validity']
        lines += [f'The reserved run ended with status `{e["status"]}` and {e["rows"]:,} recorded generations. Undefended attacks change paired strict target accuracy by {-100*g["paired_target_accuracy_drop"]:.1f} percentage points and have strict attacker-task success {percent(g["strict_attacker_success_rate"])}. The frozen attack-validity gate {"passes" if g["passed"] else "fails"}: it requires a target-accuracy drop of at least ten points or strict attacker success of at least 10%.','',
        '| Condition | Strict target correct | Strict-format valid | Published target correct | Strict attacker success | Published attacker success |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
        for key,a in e['arms'].items():
            n=a['n'];attack=key.startswith('attack/')
            strict=f'{a["strict_attacker_success"]}/{n}' if attack else '—';official=f'{a["published_attacker_success"]}/{n}' if attack else '—'
            lines.append(f'| {arm_label(key)} | {a["strict_correct"]}/{n} | {a["strict_valid"]}/{n} | {a["published_correct"]}/{n} | {strict} | {official} |')
        if e['status']=='complete':
            base=e['arms']['attack/undefended'];learned=e['arms']['attack/learned_span_filtering'];rule=e['arms']['attack/unfitted_rule_filtering']
            lines += ['',f'All {base["n"]} undefended attacked outputs remain valid under the strict target-label parser, so the observed accuracy loss is not an invalid-format count. Learned filtering changes {learned["changed_input_sequences"]} input token streams and {learned["changed_target_predictions"]} target predictions, with {learned["changed_correctness"]} correctness changes. It supplies no measured target-accuracy gain. The unfitted filter scores {rule["strict_correct"]}/{rule["n"]}, versus {base["strict_correct"]}/{base["n"]} undefended. Zero learned paired intervals describe the observed unchanged paired scores, not a universal no-effect guarantee.']
        lines += ['',f'Identical-decision filtering and two-bank outputs match in all {e["matched_equivalence_pairs"]} observed pairs. This supports implementation equivalence in this run, not a predictive advantage of the two-bank architecture.']
        if e['status']=='complete':
            lines += ['', '| Attacked arm versus undefended | Paired target-accuracy difference | Source-cluster bootstrap 95% interval |','| --- | ---: | ---: |']
            for key,a in e['arms'].items():
                if key.startswith('attack/') and key!='attack/undefended':
                    lines.append(f'| {arm_label(key.split("/",1)[1])} | {100*a["paired_accuracy_gain"]:.1f} points | {interval(a["paired_gain_ci95"])} points |')
        else:
            lines += ['', 'The protocol stops before attacked defended arms; clean-arm utility/equivalence is not an attack-efficacy estimate.']
    elif e['status']=='failed':
        lines += [f'The reserved run failed after {e["rows"]} recorded generations: `{e["error"]}`. Partial rows are retained and do not establish a completed efficacy comparison.']
    else:
        lines += [f'The reserved evaluation is `{e["status"]}`. No final efficacy conclusion is available.']
    if e['status']=='complete':
        lines += ['', '| Attacked condition | Post-hoc yes/no attacker success | Valid yes/no outputs |', '| --- | ---: | ---: |']
        for key,a in e['arms'].items():
            if key.startswith('attack/'):
                lines.append(f'| {arm_label(key.split("/",1)[1])} | {a["posthoc_yes_no_attacker_success"]}/{a["n"]} | {a["posthoc_yes_no_attacker_valid"]}/{a["n"]} |')
    if e.get('execution',{}).get('kind')=='continued_after_runtime_loss':
        x=e['execution']
        lines += ['',f'Colab replaced the first runtime. The execution-only continuation retains {x["recovered_unique_rows"]} observed rows and adds {x["new_unique_rows"]} missing planned rows, with {x["overlap_generations"]} exact overlap replay generations kept separately from the sample counts. Prompts, data, masks, model, thresholds and the original validity gate are unchanged; the native, two-bank, scoring and token-serialization function bodies are checked against the original runner. The interrupted segment has no terminal weight hash. Matching continuation start/end hashes and overlap outputs do not establish that missing check.']
    c=e.get('injected_control',{})
    if c.get('n'):
        lines += ['',f'The published injected-task instruction asks for yes/no, whereas its fixed benchmark parser expects spam/not-spam. Injected controls have {c["strict_spam_valid"]}/{c["n"]} valid outputs and {c["strict_spam_correct"]}/{c["n"]} correct under the strict spam parser. A compatible yes/no parser, audited post hoc, finds {c["posthoc_yes_no_valid"]}/{c["n"]} valid and {c["posthoc_yes_no_correct"]}/{c["n"]} correct. This secondary check does not change the frozen attack-validity gate. The injected-task control does not establish adequate spam-task competence in this prompt configuration, so low attacker-task ASR cannot establish resistance to instruction redirection. Target-accuracy damage and preservation remain separately measurable under the frozen target-drop criterion.']
    lines += ['', 'The backbone is the pinned FP32 Mamba-2.8B on a T4; recorded start/end fingerprints and cache-copy checks are audited where the run completes. Masks were precomputed on Apple MPS, so decoder/cache execution is not end-to-end routing latency. Fifty target-source clusters, one task, one prompt, one injected task and five constructors limit inference. No matched benign added-text/length control isolates instruction-specific damage from distraction by extra context. Paired intervals condition on the fixed configuration and cannot establish domain-wide or adaptive-attack robustness. Earlier three-task failures and the 2,048-token timing OOM remain failures under their original protocols.','<!-- END_FOLLOWUP_20261029 -->','']
    return '\n'.join(lines)


def report(s):
    return '# October 29 mentor follow-up\n\nThe bounded larger-embedding comparison and the separate one-task external study are recorded below. Their outcomes do not establish a general defense or a predictive benefit unique to two states. This report is generated from independently recomputed saved rows; it is not a fresh inference replication.\n\n'+paper_block(s)+'\nThe mentor\'s recommendation was useful: it produced a stronger competence baseline and tested larger sequence embeddings under explicit domain separation. BGE improves ranking modestly, while E5 gives no operating-point recall gain; neither size alone nor document flags establish useful attack removal. The next meeting should settle the paper\'s scope and author-written argument, with the evidence and limitations above retained.\n'
