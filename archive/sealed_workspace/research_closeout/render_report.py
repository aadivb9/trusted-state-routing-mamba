"""Render the final project status after all closing responses are audited."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from external_benchmark.audit_development import predictions


def main():
    result=json.loads((ROOT/'research_closeout/run_03/results.json').read_text())
    verification=json.loads((ROOT/'research_closeout/run_03/verification.json').read_text())
    assert result['status']=='complete' and verification['status']=='passed'
    recovery_path=ROOT/'research_closeout/execution_recovery.json'
    recovery=json.loads(recovery_path.read_text())
    recovery['attempt_02'].update(status='interrupted_unavailable_terminal_state',recovered_rows=572)
    recovery['attempt_02']['policy_superseded_by']='RESUME_AMENDMENT.md: retain 572 verified responses and continue only the 28 missing cases'
    recovery['attempt_02']['automatic_exports']='Local exports every 45 seconds while the collector, host and browser connection remained available'
    recovery['continuation_03']=dict(status='complete',retained_rows=572,new_rows=28,
                                    exact_overlap_controls=12,verification='run_03/verification.json',
                                    limitation='Attempt 2 terminal weight hash unavailable; resumed start/end hashes match saved initial hash',
                                    overlap_comparison=verification.get('recovery_comparison',{}))
    recovery_path.write_text(json.dumps(recovery,indent=2)+'\n')
    lines=['# Final research tests and paper close-out','',
           '**The closing test set is complete. The result supports a working cache controller with narrow routing benefits, not a demonstrated general memory-poisoning defense.**','',
           '## External benchmark: final reserved baselines','',
           'All 600 reserved baseline generations completed on frozen FP32 Mamba-2.8B on a T4. These measurements extend the failed 72-example development screen as a separately documented closing diagnostic. They do not replace that failed gate or form an attack-efficacy evaluation.','',
           '| Role | Task | Published correct | Strict correct | Strict valid | Instruction-only published correct |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    for role in ['target','injected']:
        for task in ['sst2','sms_spam','mrpc']:
            s=result['summary'][role+'/'+task]
            rows=[r for r in result['rows'] if r['role']==role and r['dataset']==task]
            empty=result['instruction_only'][rows[0]['instruction_control']]['response']
            pred=predictions(task,empty)[1]
            control=sum(r['gold']==pred for r in rows)
            lines.append(f"| {role} | {task} | {s['official_correct']}/100 | {s['strict_correct']}/100 | {s['strict_valid']}/100 | {control}/100 |")
    lines+=['','Target examples measure the legitimate task. Injected examples measure the attacker task on its own clean data; this column is not an attack success rate. Each cell is balanced and follows the original deterministic selection. Strict parsing also penalizes formatting. The official parser can count labels embedded in contradictory prose. Neither is a substitute for a demonstrated harmful attack effect.','',
            'The October 11 attempt retained 410 local responses before its runtime became unavailable. A full unchanged rerun on October 13 recovered 572 responses before a second connection loss. The final table retains those 572 responses and adds only the 28 missing cases from a documented continuation. Twelve overlap examples and every instruction-only control reproduced identical token outputs before continuation. Each case counts once; controls and earlier attempts are not independent replications. The interrupted segment\'s terminal weight hash is unavailable. Its initial hash matches both the resumed process\'s initial and final hashes. See RESUME_AMENDMENT.md for the execution-only amendment.','',
            'The frozen router accepted all 4,500 prepared attack inputs and all 600 legitimate baseline inputs. Every routing question had an unsupported attribute. This shows absent rejection coverage under the specified trust boundary. **No attack-conditioned generations were run for those 4,500 inputs.** The original security-evaluation gate remained closed.','',
            '## Final checks','',
            '- 26 existing CPU tests passed.',
            '- Independently recomputed 1,520 historical candidate predictions and 1,520 free-generation scores, 44 summary cells, six paired means, and the headline +42 to +62 percentage-point interval.',
            '- Rechecked the published-HiSPA saved result: 576 candidate predictions and 128 matched-length controls.',
            '- External preparation matches 600 official source selections, 4,500 attack strings and 672 task instructions.',
            '- The earlier 72-example competence result was independently rescored and its pre-execution seal verified.',
            '- All 600 new baseline responses and six summary cells were independently rescored; all 11 closing source/input hashes match.',
            '- All observed 2.8B checkpoint hashes match the successful T4 trial, including the continuation start/end. The interrupted segment lacks a terminal hash; this missing check is not represented as performed. No model or router training occurred.',
            '', 'Saved-output audits are not new GPU replications. The 600-example closing diagnostic contains new GPU measurements completed across the documented execution segments; the existing T4 feasibility and earlier attack experiments are retained measurements.','',
            '## Paper conclusion','',
            'Context-aware text routing improved attacked candidate ranking from 28% to 80% inside a controlled fact schema. That result matters because it establishes useful evidence checking under the stated assumptions. It does not establish a unique advantage of two banks: identical-decision filtering matched evaluated outputs. The failure to transfer to unfamiliar attributes, the all-accept external router, and weak external clean-task performance limit the security claim. The published-trigger pilot did not establish baseline harm, so its unchanged defended accuracy is not evidence of protection.','',
            'The research closes with these positive and negative findings. Further model selection, router development and full official security benchmark reproduction are not required to describe this completed study; they remain unperformed work and cannot appear as results.','',
            '## Paper package','',
            '- [Full editable manuscript](../paper/manuscript.md)',
            '- [Review PDF](../paper/manuscript.pdf)',
            '- [Claim-to-evidence mapping](../paper/evidence_ledger.md)',
            '- [Editorial submission items](../paper/README.md)',
            '- [Final raw answers](run_03/results.json) and [independent audit](run_03/verification.json)',
            '- [Frozen closing protocol](PROTOCOL.md), [resume amendment](RESUME_AMENDMENT.md), [continuation notebook](final_diagnostic_resume.ipynb), and [execution recovery](execution_recovery.json)',
            '', 'Remaining submission work concerns author details, venue formatting, coauthor review, disclosures and artifact-release decisions. No submission or public release has been performed.']
    (ROOT/'research_closeout/REPORT.md').write_text('\n'.join(lines)+'\n')
    target={task:result['summary']['target/'+task]['official_correct'] for task in ['sst2','sms_spam','mrpc']}
    plain=f'''# What this research established

**The memory-backup mechanism works. The current system is not a demonstrated general defense against memory poisoning.**

The idea was to save a trusted memory state, test incoming information in a second state, and keep or discard the update. That mechanism passed implementation checks, including on Mamba-2.8B running in FP32 on a T4.

The best result was a rise from 28% to 80% attacked accuracy on a controlled fact task. This was two-choice answer ranking, not unrestricted generation. It shows that checking claims against known facts can help within the router's supported schema.

The improvement disappeared on unfamiliar abstract attributes. Filtering out the same rejected text matched the tested two-bank outputs, so the evidence does not show an accuracy benefit unique to storing two memories.

We also tested real externally sourced benchmark data. The final clean target scores were {target['sst2']}/100 for sentiment, {target['sms_spam']}/100 for spam, and {target['mrpc']}/100 for paraphrase detection under the published scorer. The frozen router rejected none of 4,500 published attack inputs. That is a coverage failure, not a measurement that all 4,500 attacks succeeded: those attack-conditioned generations were not run after the clean-task gate failed.

**The research is worth reporting as a study of benefits and limits. It does not justify claiming that dual-state memory solves poisoning.** The final paper retains the positive result, failed transfer, filtering controls, hardware success and external-benchmark limitations. Remaining work is manuscript review and submission preparation, not another research cycle.
'''
    (ROOT/'paper/plain_language_summary.md').write_text(plain)
    p=ROOT/'research_closeout/STATUS.md';s=p.read_text().replace('Unchanged rerun in progress after original runtime loss; automatic local checkpoint exports enabled','Completed as 572 retained + 28 continued cases; 12 overlap checks; missing interrupted terminal hash disclosed').replace('Full Markdown, HTML and review PDF generated; final baseline table pending','Full Markdown, HTML and review PDF include verified final baseline results')
    p.write_text(s)
    p=ROOT/'paper/README.md';s=p.read_text().replace('The closing 600-example diagnostic is separate from the original failed development gate. Its pending status remains explicit until completed results are verified.','The closing 600-example diagnostic is complete and independently verified. It is separate from the original failed development gate.')
    p.write_text(s)
    p=ROOT/'external_benchmark/README.md';s=p.read_text()
    note='**Closing update:** the separate 600-example reserved-baseline diagnostic is complete and independently verified. The failed development gate remains recorded; 4,500 attack inputs received router coverage checks, not attack-conditioned generations. See [the final close-out report](../research_closeout/REPORT.md).\n\n'
    if note not in s:s=s.replace('# External benchmark preparation\n\n','# External benchmark preparation\n\n'+note)
    p.write_text(s)
    p=ROOT/'paper/evidence_ledger.md';p.write_text(p.read_text().replace('and `verification.json` when complete','and `verification.json`'))
    print('Wrote closing report and final status')


if __name__=='__main__':
    main()
