"""Apply the audit revision to the active manuscript, leaving the archive intact."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'paper/manuscript.md'
    text=path.read_text()
    if '<!-- AUDIT_REVISION_20261022 -->' in text:raise RuntimeError('Revision already applied; edit deliberately rather than reapply')
    first=text.index('## Abstract');end=text.index('## 1. Introduction')
    abstract='''## Abstract

Preserving an earlier recurrent state permits discarding an untrusted update, but its security value depends on which updates are accepted. We examine a two-bank cache controller in frozen Mamba, with external text routing and identical-decision filtering controls. On an independently generated synthetic benchmark within a restricted fact schema, attacked two-candidate accuracy rises from 28% to 80%, whereas strict free-generation correctness rises from 20% to 28%. An unfitted only-conflict rule using the same hand-built extractor matches every trained-cascade decision on the 200 observed notes; the test therefore isolates neither a learned-routing contribution nor an accuracy benefit unique to two banks. Transfer to unfamiliar abstract attributes fails. Published-trigger controls use an easy restricted-choice adaptation and establish no defense effect. A fixed embedding comparison also transfers poorly: aggregate validation constraints conceal email false-positive rates of 50% for the expanded bigram head and 20% for the expanded embedding head. A post-audit conservative router closes the reported acceptance bypasses but abstains on 47 of 100 benign synthetic notes and every observed external input. These are engineering regressions and observed-set diagnostics, not new generalization evidence. Fixed external clean-competence configurations fail, and no external attack-conditioned efficacy result is available. Cache mechanics pass prior T4 feasibility checks and a new pinned small-model CPU reference comparison, but neither establishes general speed or security advantages. The findings support a bounded study of cache implementation and task-specific evidence checking, while showing that transferable, useful poisoning defense remains unestablished.

'''
    text=text[:first]+abstract+text[end:]
    text=text.replace('Our contribution is an empirical examination of an implemented cache controller and the boundaries of its observed benefit.',
                      'Our contribution is an empirical examination of an implemented cache controller and the boundaries of its observed benefit. The post-hoc unfitted-rule comparison leaves an incremental contribution from fitting the symbolic router unestablished; the filtering controls likewise leave a two-bank predictive contribution unestablished.')
    methods='''### 3.5. Post-audit conservative policy and calibration safeguards

The engineering revision `guarded-symbolic-v1` is an unfitted policy, separate from all historical routers. It produces accept, reject, or abstain with a reason and no calibrated probability claim. Unsupported questions, absent or ambiguous trusted values, and unverifiable notes abstain. Supported question paraphrases and number words are normalized within the same limited attribute schema. Operative answer/source overrides and extracted conflicts reject; explicitly rejected quotations are distinguished from unscoped quotations. Only acceptance commits a candidate state; rejection and abstention preserve the trusted state. Empty updates leave it unchanged. This policy does not add a general entailment model or validate every clause of an accepted document.

The prospective calibration helper requires enough independent benign source groups in every represented domain, a minimum source-balanced attack recall, and a one-sided exact binomial upper bound on each domain's benign-group rejection rate. Repeated benign presentations from a source count as one group; any rejected benign presentation flags that group. The default minimum is 100 benign groups per domain, recall at least 0.5, and an upper FPR bound at most 0.05 with Bonferroni allocation of family alpha 0.05 across specified domains. No feasible useful threshold and insufficient calibration support are explicit outcomes. These software safeguards have regression tests; they have not been used to refit or replace the historical frozen operating points, and do not guarantee calibration on unseen domains.

'''
    text=text.replace('## 4. Experimental design',methods+'## 4. Experimental design')
    design='''### 4.5. Audit calculations, annotation repair, and prospective work

Post-audit calculations reuse the observed records and are descriptive. We compare the trained cascade, fast-only head, detailed-extractor only-conflict rule and conservative revision on the same 200 notes. The exact audit bypasses are regression inputs, not held-out attacks. Paired source-group bootstrap uses the same 5,000 draws and fixed seed separately for ranking and strict generation. Intervals condition on this schema, provider and template collection and are not adjusted across all exploratory study comparisons.

Frozen MiniLM inference was repeated in the recorded CPU software environment without fitting heads or changing thresholds. All six heads' per-domain validation scores were reconstructed; final embedding probabilities were compared with the saved values. A new review exporter randomizes candidate positions, shuffles notes and uses opaque identifiers. Its reviewer packet excludes the mapping, labels and seed. Preparation of that packet does not constitute an independent review; a returned independent review and human spot-check remain pending.

A separately frozen prospective external protocol selects new exact-normalized-text-disjoint records from the pinned published sources and uses unchanged published attack constructors. It has a new 72-example competence screen, reserved clean baselines and 4,500 attack presentations. Its hash-order balanced selection and fixed trailing-newline prompt are adaptations, not the original first-N procedure or full official benchmark reproduction. No new GPU execution occurred: local CUDA preflight failed. The protocol requires all tasks to pass the unchanged competence gate before any attack-conditioned generations, with no prompt sweep, checkpoint replacements or selection of favorable tasks. Same-task injection rows must be distinguished when interpreting attack objectives. These prepared inputs and runners are not efficacy results.

'''
    text=text.replace('## 5. Results',design+'## 5. Results')
    result=json.loads((ROOT/'experiments/15_router_remediation/results.json').read_text())
    ml=json.loads((ROOT/'experiments/15_router_remediation/embedding_reconstruction.json').read_text())
    cpu=json.loads((ROOT/'experiments/18_runtime_validation/cpu_results.json').read_text())
    lines=['### 5.7. Post-audit baselines, routing failures and safeguards','',
           '<!-- AUDIT_REVISION_20261022 -->','',
           'The unfitted only-conflict rule, trained cascade and fast-only head make identical decisions on all 200 observed synthetic notes. They each reject 99/100 malicious notes and 0/100 benign notes. The rule uses the existing detailed hand-built extractor and was evaluated post hoc; it is not a schema-free detector or a new confirmatory result. Identical downstream state selection under the same decisions is a deterministic inference, not a fresh rule-baseline generation run.','',
           '| Metric on the main synthetic test | Original | Controller | Paired difference | 95% interval |',
           '| --- | ---: | ---: | ---: | ---: |',
           '| Two-choice ranking | 28/100 | 80/100 | 52 percentage points | [42, 62] points |',
           f'| Strict free generation | 20/100 | 28/100 | 8 percentage points | [{result["paired_free"]["ci95"][0]*100:.0f}, {result["paired_free"]["ci95"][1]*100:.0f}] points |','',
           'The free-generation interval includes zero, and clean strict generation is 28/100. The ranking gain cannot be substituted for a general-generation effect.','',
           'Frozen-weight analysis bounds unsupported-question risk below 0.05862 in the fast head and 0.05064 in the detailed head, both below the 0.5 threshold. Thus this historical router deterministically accepts unsupported questions under the inspected feature constraints. Spelled-out false counts, untyped numeric overrides, mixed agreement plus overrides, and supported-meaning question paraphrases produce further accepted malicious-note probes. These are CPU routing bypasses, not measured successful language-model attacks.','',
           '| Router on observed notes | Attack reject | Attack abstain | Benign accept | Benign abstain |',
           '| --- | ---: | ---: | ---: | ---: |']
    for key,label in [('cascade','Trained cascade'),('fast_only','Fast-only head'),('only_conflict_rule','Unfitted only-conflict'),('guarded_v1','Conservative revision')]:
        s=result['router_comparison'][key]
        lines.append(f'| {label} | {s["malicious"]["reject"]}/100 | {s["malicious"]["abstain"]}/100 | {s["safe"]["accept"]}/100 | {s["safe"]["abstain"]}/100 |')
    lines+=['','The conservative revision accepts none of the six audit probes or 100 observed malicious notes. However, it detects only 60 of those notes by rejection and abstains on the other 40; it also abstains on 47/100 benign notes. All 160 abstract-note presentations and all 5,100 external coverage presentations abstain. Abstention prevents those updates from being committed; it does not establish semantic attack detection, transferred competence, or preservation of task utility. No revised-controller answer generations were run.','',
            '### 5.8. Per-domain validation reconstruction','',
            'The following counts use each original frozen threshold. Synthetic validation contains 100 benign notes; email contains only ten benign source records. Aggregate benign FPR can therefore conceal large email errors.','',
            '| Fitting corpus | Representation | Synthetic benign rejected | Email benign rejected | Aggregate benign rejected |',
            '| --- | --- | ---: | ---: | ---: |']
    for key,model in ml['validation'].items():
        regime,representation=key.split('/');domain={r['domain']:r for r in model['domains']}
        a,b=domain['synthetic'],domain['email'];label='Synthetic' if regime=='synthetic_only' else 'Synthetic + email'
        lines.append(f'| {label} | {representation} | {a["benign_rejected"]}/{a["benign_n"]} | {b["benign_rejected"]}/{b["benign_n"]} | {a["benign_rejected"]+b["benign_rejected"]}/110 |')
    lines+=['','The expanded bigram head rejects 5/10 benign emails (50%) while meeting its aggregate constraint at 5/110. The synthetic-only and expanded embedding heads reject 4/10 (40%) and 2/10 (20%) benign emails. This is measured development miscalibration, not merely a possible limitation or a reason to recalibrate against observed final outcomes.','',
            'The repeated pinned CPU encoder reproduces all 2,100 stored final embedding probabilities exactly, with zero decision differences. Short-text encoder parity has maximum error 2.98e-8 and all four embedding harness tests pass in the recorded ML environment. These are computational repetitions on the same records and frozen heads, not additional independent samples or new fitting.','',
            '### 5.9. Sample accounting and reference execution','',
            '| Study | Presentations or decisions | Source units | Reuse and scope |',
            '| --- | ---: | --- | --- |',
            '| Main synthetic test | 200 notes; 100 attacked pairs | 100 source groups | Generated within one supported schema |',
            '| Abstract transfer | 160 note presentations | 32 abstract tasks | Constructed notes; source-group analysis |',
            '| HiSPA interventions | 112 evaluations | 16 reused cases | Seven conditions on observed cases |',
            '| Router transfer | 1,050 presentations; 6,300 decisions | 150 records; 134 source groups | Six models share the same inputs; payload families reused |',
            '| Instruction-tuned screen | 72 examples | Previously observed development records | Gate failure retained |','',
            f'A fresh pinned Mamba-130M CPU comparison captures unpatched full-sequence reference logits before installing the cache wrapper. At 8-, 16- and 32-token prefixes with four-token suffixes, maximum full-reference logit error is {max(r["reference_max_error"] for r in cpu["checks"]):.5g} and maximum chunk/token error is {max(r["chunk_token_max_error"] for r in cpu["checks"]):.5g}, below 1e-3. Cloned states have independent storage, committed states remain unchanged during candidate advancement, and before/after weight hashes match the recorded historical checkpoint. This is a small-model CPU smoke test, not a new T4 timing or large-model efficacy result.','',
            'Local input/code seals establish the checked artifact sequence, not externally authenticated preregistration. Source-group intervals do not capture all uncertainty from shared providers, domains or attack families.','']
    text=text.replace('## 6. Discussion','\n'.join(lines)+'\n## 6. Discussion')
    text=text.replace('The model-based semantic reviewer always saw the correct candidate first and notes in a fixed order, creating annotation-review cues;',
                      'The historical model-based semantic reviewer always saw the correct candidate first and notes in a fixed order, creating annotation-review cues. The replacement exporter removes these mechanical cues, but no returned independent review or human spot-check is available;')
    text=text.replace('The abstract pilot uses constructed cloze questions and unauthenticated supplied abstracts.',
                      'The abstract pilot uses constructed cloze questions and a supplied source snapshot. A later provenance audit matches all 120 abstract byte strings, titles and original question labels to the cited Git commit; this does not independently authenticate every underlying publication.')
    text=text.replace('The artifacts retain exact model revisions, input/code hashes, raw predictions, timing records, data provenance, notebook reproductions and observed weight-hash checks.',
                      'Later artifacts and the revised replay loader pin model/tokenizer revisions; earlier Mamba-130M loaders omit a revision argument. Historical fingerprints and later pinned records help detect drift but do not make those earlier loaders pinned. Artifacts retain input/code hashes, raw predictions, timing records, data provenance, notebook reproductions and observed weight-hash checks. The third-party instruction-checkpoint conversion remains unvalidated against native upstream weights. The new pinned CPU parity check does not close that conversion question or establish GPU/optimized-kernel equivalence.')
    conclusion='''Two-bank state preservation is feasible under the measured frozen-Mamba workloads, but the observed synthetic gain is explained equally well by an unfitted rule using the same hand-built extractor, and identical-decision filtering matches recorded controller outputs. Learning and two-bank state storage therefore have no isolated predictive contribution in this evidence. The conservative revision replaces silent unsupported acceptance with explicit abstention, at substantial benign-coverage cost. Embedding replay confirms the saved results and exposes measured domain calibration errors. Neither this engineering repair nor the prepared prospective external protocol establishes useful transferable poisoning defense. The defensible contribution is a reproducible implementation and empirical account of its benefits, shortcuts, failures and unresolved validation requirements.'''
    a=text.index('## 8. Conclusion');b=text.index('## References',a)
    text=text[:a]+'## 8. Conclusion\n\n'+conclusion+'\n\n'+text[b:]
    path.write_text(text)
    print('Updated active manuscript; historical manuscript untouched')


if __name__=='__main__':main()
