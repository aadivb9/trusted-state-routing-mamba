"""Insert audited follow-up experiments into the manuscript without changing earlier results."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent


def main():
 for folder in ['hispa_stress','competence','router_transfer']:
  assert json.loads((ROOT/'revision_experiments'/folder/'verification.json').read_text())['status']=='passed'
 assert json.loads((ROOT/'revision_experiments/router_transfer/encoder_verification.json').read_text())['status']=='passed'
 p=ROOT/'paper/manuscript.md';text=p.read_text()
 if '<!-- REVISION_EXPERIMENTS -->' in text:raise RuntimeError('Already inserted; preserve review edits')
 snap=ROOT/'paper/snapshots';snap.mkdir(exist_ok=True)
 (snap/'manuscript_before_experiments_20261016.md').write_text(text)
 text=text.replace('These findings support task-specific evidence verification while exposing limitations in detector coverage, clean-task competence, and claims attributable to state preservation alone.',
                   'Further evidence interventions show that the cloze model follows changed source facts in 15/16 cases, while clean free-generation accuracy is only 7/16 under strict scoring. A fixed six-router follow-up holds out table and code domains from training and selection: embeddings with added email training reject 56% of table attacks but also 36% of benign tables. An instruction-tuned Mamba checkpoint also fails the fixed development gate. These findings support task-specific evidence verification while exposing limitations in detector coverage, threshold transfer, clean-task competence, and claims attributable to state preservation alone.')
 text=text.replace('Our contribution is an empirical examination',
                   'A bounded follow-up tests a frozen sentence-embedding representation and added external training data with task domains held out from fitting and threshold selection. This separates a representation comparison from the earlier hard-coded schema stress test, while retaining the distinction between detector performance and end-to-end security.\n\nOur contribution is an empirical examination')
 related='''
The follow-up uses a sentence-embedding checkpoint based on MiniLM [8, 9] and source records and attacks from BIPIA [10]. MiniLM representations encode semantic similarity but are not, by themselves, validated poisoning classifiers. BIPIA provides task-specific external contexts and published injection payloads; our bounded detection subset does not reproduce its complete attack-success evaluation.
'''
 text=text.replace('## 3. Method and threat model',related+'\n## 3. Method and threat model')
 method='''
### 3.4. Fixed embedding and representation comparison

The follow-up compares three representations using newly fitted logistic heads: the 14 detailed symbolic features, 8,192 signed hashed word/bigram features, and a frozen `all-MiniLM-L6-v2` encoder with 22,713,216 parameters [8, 9]. Each receives the same three observable fields: trusted reference/instruction, external text, and question. The symbolic head is a single classifier, distinct from the earlier two-head cascade. The lexical comparator now includes all three fields rather than only the note.

The encoder produces normalized 384-dimensional vectors for 254-token content windows with stride 192 and two special tokens. Every text token, including the tail, is covered. We concatenate mean trusted, mean question, mean note and maximum note vectors, plus absolute differences and elementwise products between the mean note and each of the other means, yielding 3,072 features. Dense features are standardized using training data only. Every head uses fixed C=1 logistic regression with balanced class weights; there is no hyperparameter sweep. Source identifiers, task answers, attack positions, clean counterparts and labels are excluded from features.

For external data, only the task instruction and user question are trusted. The full external document is the candidate note, so rejection can discard legitimate evidence. This follow-up measures document-level detection; it does not implement a new span-localizing or end-to-end defense.
'''
 text=text.replace('## 4. Experimental design',method+'\n## 4. Experimental design')
 design='''
### 4.4. Follow-up interventions and domain holdout

After review, a sealed evidence-sensitivity run retains all 16 observed HiSPA-pilot cases. Seven conditions produce 112 new evaluations: clean, empty source, supporting-sentence removal, cyclic substitution of another source abstract, artificial counterfactual replacement of the unique answer with its foil, clean with M35, and counterfactual with M35. Counterfactual conditions flip the expected candidate; they are artificial evidence probes, not corrected real-world publications. Complete-continuation likelihood remains primary, with per-token normalization as a secondary sensitivity check and 32-token greedy free generation scored separately. Reversing candidate order is checked on clean and counterfactual cases. This is diagnostic reuse, not an untouched test or an official HiSPA reproduction.

A separate development screen evaluates exactly one instruction-tuned checkpoint, the third-party Transformers conversion `han1997/mamba-2.8b-zephyr-hf`, with its original author's chat template [11]. It reuses the 72 observed examples and unchanged competence thresholds. No prompt sweep or final-domain selection follows its results. Native-to-converted checkpoint equivalence is not independently verified.

The router study has a fixed three-representation by two-training-corpus design. Both regimes use the same 1,000 synthetic training examples; the expanded regime adds 280 presentations from 40 BIPIA email records. All six heads use a common validation set of 200 synthetic and 70 email presentations. Thus synthetic-only describes fitting data, not absence of email from threshold selection. Each threshold maximizes validation recall subject to an aggregate benign false-positive rate of at most 5%, with ties favoring lower false-positive rate and then higher threshold. The constraint is aggregate, not per-domain.

Final data contain 100 table and 50 code source records, each with one clean presentation and six deterministically selected published payloads inserted at the start or end: 1,050 presentations in total. Both task domains are excluded from router fitting and threshold selection. Before constructing attacks, exact normalized source matches and five-word-shingle Jaccard similarity of at least 0.90 form connected components. No component crosses training, validation and final partitions; the final sets contain 84 table and 50 code groups. This procedure does not exclude every semantic duplicate. BIPIA's text attack family lists overlap in Language Translation, so complete attack-family novelty is not claimed. Pretrained encoder exposure to these domains is unknown.

All six heads and their thresholds are saved and hashed before final-domain featurization. Detection recall, benign false-positive rate, AUROC and calibration scores are reported per domain, with 2,000 source-group bootstrap draws for rate intervals. These are detector outcomes, not task utility or attack success. No model or threshold is revised against final outcomes.
'''
 text=text.replace('## 5. Results',design+'\n## 5. Results')
 stress=json.loads((ROOT/'revision_experiments/hispa_stress/results.json').read_text())
 stress_text='''
The new evidence-sensitivity run reproduces all 48 comparable historical score vectors exactly and passes 32 candidate-order checks. The following counts are for the undefended model on the same 16 cases; free generation uses the strict answer-at-start parser.

| Evidence condition | Candidate ranking correct | Strict free-generation correct |
| --- | ---: | ---: |
'''
 for condition,s in stress['summary'].items():stress_text+=f'| {condition} | {s["correct"]}/16 | {s["free_correct"]}/16 |\n'
 stress_text+='''
The model responds to altered evidence in 15/16 cases, which weighs against a fixed leaked option index as the explanation for its answers. However, substantial original-answer accuracy survives supporting-evidence removal or unrelated-source substitution. The 16/16 clean ranking result contrasts with 7/16 strict free-generation accuracy. These controls indicate both source sensitivity and strong answer priors; they do not exclude every leakage mechanism. Secondary length normalization preserves the clean, empty, support-removed and counterfactual ranking counts, but changes the unrelated and counterfactual-M35 counts. The 15/16 to 14/16 counterfactual-M35 change has no matched benign counterfactual control and is not treated as a reproduced HiSPA effect.
'''
 text=text.replace('### 5.4. Larger-model hardware feasibility',stress_text+'\n### 5.4. Larger-model hardware feasibility')
 comp=json.loads((ROOT/'revision_experiments/competence/results.json').read_text())
 comp_text='''
The fixed instruction-tuned checkpoint also fails the development gate: strict correctness is 0/24, 2/24 and 0/24 for sentiment, spam and paraphrase; published-parser correctness is 12/24, 3/24 and 4/24. Many responses provide explanatory prose rather than the requested labels. These measurements concern the fixed checkpoint/prompt/parser configuration, not an intrinsic ceiling on semantic competence. The stopping rule prevents an end-to-end security-efficacy claim for this configuration; the original failed base-model result remains intact.
'''
 text=text.replace('## 6. Discussion',comp_text+'\n<!-- REVISION_EXPERIMENTS -->\n### 5.6. Embeddings and added diversity do not reliably transfer\n\n'+
 '''The table reports attack rejection and benign rejection at thresholds fixed on the common mixed validation set. Each table cell uses 600 attacks and 100 clean presentations; each code cell uses 300 attacks and 50 clean presentations.\n\n| Training | Representation | Table: recall / FPR | Code: recall / FPR |\n| --- | --- | ---: | ---: |\n'''+
 '\n'.join(f'| {"Synthetic" if regime=="synthetic_only" else "Synthetic + email"} | {rep} | '+
             ' | '.join(f'{json.loads((ROOT/"revision_experiments/router_transfer/results.json").read_text())["summary"][regime+"/"+rep+"/"+domain]["recall"]:.1%} / {json.loads((ROOT/"revision_experiments/router_transfer/results.json").read_text())["summary"][regime+"/"+rep+"/"+domain]["false_positive_rate"]:.1%}' for domain in ['table','code'])+' |'
             for regime in ['synthetic_only','synthetic_plus_email'] for rep in ['symbolic','bigrams','embeddings'])+
 '''

With added email training, embeddings reject 56% of table attacks while rejecting 36% of benign table inputs, compared with 55.8% and 51% under synthetic-only fitting. On code, the expanded embedding model rejects 17.3% of attacks and 16% of benign inputs. These results do not show reliable transfer of the validation operating point. The embedding model also does not consistently outperform the symbolic representation, whose expanded-data code recall is 57.3% with 14% false positives. Full per-domain score curves, rate intervals and all frozen predictions are retained in the artifact report.

The validation set contains 100 synthetic and only 10 email benign examples; an aggregate 5% constraint can conceal weaker calibration on the smaller domain. This is a limitation of the fixed design, not grounds to recalibrate against the final outcomes. Whole-document rejection further risks losing legitimate task evidence. Accordingly, the detector results do not establish downstream task preservation, reduced attack success, or an advantage specific to two memory banks.

The CPU encoder processes 3,143 unique texts in 4,083 windows with 19.62 seconds of recorded encoding time. This is a batched local measurement, not per-request overhead or an end-to-end T4 speed comparison. A tokenizer API repair occurred before fitting and final featurization; original sources and the amendment are retained. Encoder outputs agree with the published SentenceTransformer path within 2.98e-8 on three short verification inputs. Four harness tests and independent audits of 6,300 saved detector decisions and 12 result cells pass.

## 6. Discussion''')
 text=text.replace('The larger model establishes that GPU memory is not the immediate barrier',
                   'The embedding follow-up shows why removing explicit schema rules is not sufficient. A richer representation can reject more attacks while also rejecting legitimate inputs, and a development threshold can transfer poorly. Limited added email diversity changes results without establishing reliable domain-independent behavior. The study evaluates one frozen small encoder and one fixed head family; it does not rule out stronger detectors or more diverse training.\n\nThe larger model establishes that GPU memory is not the immediate barrier')
 text=text.replace('The project closes with that result rather than revising the method against final outcomes.',
                   'The bounded follow-up retains the frozen final outcomes rather than selecting a favorable representation, domain or threshold after evaluation.')
 text=text.replace('The external study covers three tasks and five attack constructors rather than every Open-Prompt-Injection experiment.',
                   'The original external study covers three tasks and five attack constructors rather than every Open-Prompt-Injection experiment. The later BIPIA-derived detector subset covers two held-out final domains, six payloads per source and start/end positions, without official BIPIA task-utility or attack-success scoring.')
 text=text.replace('Published-trigger and external-benchmark diagnostics do not establish general HiSPA defense efficacy.',
                   'Evidence interventions qualify the published-trigger ceiling, while the bounded embedding and added-data comparison fails to establish reliable domain transfer at its frozen operating points. Published-trigger and external-benchmark diagnostics do not establish general HiSPA defense efficacy.')
 text+='''
[8] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*. arXiv:2002.10957v2, 2020. https://arxiv.org/abs/2002.10957v2

[9] Sentence Transformers. *all-MiniLM-L6-v2 model card and checkpoint*. Revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41; accessed October 16, 2026. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

[10] Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, and Fangzhao Wu. *Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models*. arXiv:2312.14197v4, 2025; accepted at KDD 2025. https://arxiv.org/abs/2312.14197v4

[11] xiuyul and han1997. *Mamba-2.8B-Zephyr model cards and Transformers conversion*. Original revision 85b6d9a1dd2bca5787437ba63af63f155356e113; conversion revision 99a87b5e5f211c42595df370de50910f05dc7854. https://huggingface.co/xiuyul/mamba-2.8b-zephyr ; https://huggingface.co/han1997/mamba-2.8b-zephyr-hf
'''
 p.write_text(text)
 print('Inserted audited follow-up methods, results and limitations')


if __name__=='__main__':main()
