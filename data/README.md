# Data availability and acquisition

Raw corpora and abstract/reviewer packets are not redistributed in this public source export. Dataset terms differ from code licenses. Public provenance files retain the source locations, pinned commits and expected hashes:

- [Open-Prompt-Injection task download manifest](../archive/sealed_workspace/external_benchmark/raw/download_manifest.json): SST-2, UCI SMS Spam and MRPC source URLs/hashes.
- [Open-Prompt-Injection implementation manifest](../archive/sealed_workspace/external_benchmark/upstream_manifest.json), commit `fbed5cca347488f88363790ae3ae97c56116da6a`.
- [BIPIA provenance](../archive/sealed_workspace/revision_experiments/router_transfer/raw/provenance.json), commit `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`. Its [license notice](../archive/sealed_workspace/revision_experiments/router_transfer/raw/LICENSE) identifies dataset-specific terms.
- HiSPA/RoBench source: [TortueSagace/hispa](https://github.com/TortueSagace/hispa/tree/898d7f7b78ba8f934f70340d12e9c1183bb623e5). The public controller downloads only the extractor when called, checks SHA-256 `e1ac82ae8ab11b35613396eb8fd34f3b4b987343cd470da7e52e7edccfb99e48`, and does not claim upstream redistribution rights.

The preparation and evaluation code is included for review. Running the full historical pipeline needs the exact source inputs and their applicable terms, plus the sealed artifacts that are absent here. Full internal audits remain available in the author's local mentor package; the public check has a deliberately narrower, disclosed scope.

Private reviewer mappings, shuffle seeds and raw annotation request/response packets remain private. Public summary numbers do not substitute for human annotation validation.

October 29 adds BIPIA summarization/XSum-derived contexts and new SST-2/SMS selections. Raw corpora, model responses, masks and reversible generated IDs are omitted; public derived observations support count checks only. Dataset access does not imply unrestricted redistribution.
