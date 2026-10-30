# Attribution and public-release scope

This release is publicly viewable research code. No blanket reuse license is assigned to the first-party work by this publication. Third-party code retains its own license; availability of this repository does not replace dataset-specific terms.

## Included third-party implementation copies

Open-Prompt-Injection, Yupei Liu: <https://github.com/liu00222/Open-Prompt-Injection>, pinned commit `fbed5cca347488f88363790ae3ae97c56116da6a`. Its MIT notice is retained at `archive/sealed_workspace/external_benchmark/upstream/LICENSE`. This covers included implementation/view copies, not an assertion about the underlying task corpora.

BIPIA, Microsoft: <https://github.com/microsoft/BIPIA>, pinned commit `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`. Its notice is retained at `archive/sealed_workspace/revision_experiments/router_transfer/raw/LICENSE`; the notice separately identifies dataset terms. Raw benchmark data are omitted.

HiSPA, Alexandre Le Mercier and collaborators: <https://github.com/TortueSagace/hispa>, pinned commit `898d7f7b78ba8f934f70340d12e9c1183bb623e5`. No repository license was exposed at release inspection. The embedded extractor is not redistributed in this export; a pinned, checksum-verified download is used when extraction is invoked. The paper provides the research citations.

Model weights are not included. Model/checkpoint use is governed by each provider's terms. The paper and generated figures are the current research draft, with its stated empirical limits.

October 29 sequence encoders: BAAI/bge-base-en-v1.5 and intfloat/e5-base-v2 are referenced at pinned revisions; weights are not distributed. BIPIA summarization uses XSum/BBC-derived text locally. No XSum or new SST-2/SMS source text is redistributed in this update.
