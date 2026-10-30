# Paper discussion for the next mentor meeting

The paper should center on a bounded research question: when does rejecting an untrusted recurrent-state update preserve useful answers, and does keeping two state banks add anything beyond applying the same text decisions before inference?

A defensible structure is:

1. State the deployment boundary. Trusted instructions and demonstrations differ from untrusted external text. Explain what a persistent poisoned state means in this implementation and how it differs from ordinary prompt injection into a fresh context. The current experiments do not cover a complete long-lived agent memory system.
2. Describe cache cloning, speculative update, commit/rollback and unchanged backbone weights. Show the filtering equivalence control alongside the mechanism, rather than treating it as a minor ablation.
3. Use the restricted synthetic result to establish a conditional capability. Explain why the supplied fact schema makes the router's job easier and why the unfitted rule matters.
4. Report the negative transfer and HiSPA controls as limits of that capability. Keep restricted-choice scores separate from free generation.
5. Present the larger sequence-encoder comparison with calibration-before-final selection and the new summarization holdout. Lead with payload removal and clean false positives; AUROC is supporting evidence. Explain that encoder family/objective and parameter count change together.
6. Present the one-task SST-2 study with development, confirmation and reserved targets visibly separated. Compare clean utility, attacked utility, unfitted filtering and matched learned-filter/two-bank outputs. Preserve all failed historical protocols as distinct studies. If the injected-task control is weak, distinguish target-task damage from successful execution of the attacker's task and do not interpret low ASR as robustness.
7. Discuss what the results support, what they fail to show, and which limitations are intrinsic to the experiment. Avoid claiming a general defense, a learned benefit explained by rules, or an architectural accuracy gain reproduced by filtering.

Before submission, the authors and mentor should rewrite the motivation and argument in their own voice, confirm novelty against related work, decide the target venue, resolve authorship, and check dataset/citation obligations. Model-based annotation disagreements are not human adjudication. Native instruction-checkpoint conversion remains unverified. The failed full timing experiment cannot support an unconditional efficiency claim.

This is an author revision plan, not a statement that the manuscript is submission-ready or that a meeting has been arranged.

The reserved attack study does not include a matched benign added-text/length control. Interpret its accuracy drop as target disruption under the chosen published constructors, without attributing the entire drop specifically to instruction execution. The weak injected-task control is a separate limitation.
