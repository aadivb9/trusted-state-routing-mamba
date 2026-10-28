# Implementation map

| Module | Responsibility |
| --- | --- |
| [memory/state_controller.py](memory/state_controller.py) | Cache capture, copying, continuation, selection and generation |
| [routing/context_router.py](routing/context_router.py) | Original symbolic context features and routing |
| [routing/guarded_router.py](routing/guarded_router.py) | Conservative accept/reject/abstain policy |
| [routing/embedding_router.py](routing/embedding_router.py) | Frozen encoder and detector comparison |
| [validation/calibration.py](validation/calibration.py) | Domain support, false-positive bounds and useful-recall requirements |
| [validation/blinded_review_v2.py](validation/blinded_review_v2.py) | Private-entropy presentation randomization and opaque IDs |
| [runtime/serialization_v2.py](runtime/serialization_v2.py) | Full-prompt token serialization and boundary ownership |
| [runtime/pinned_models.py](runtime/pinned_models.py) | Recorded checkpoint/revision pairs |
| [evaluation/published_task_scoring.py](evaluation/published_task_scoring.py) | Strict and permissive task scoring |

Public source views are ordinary materialized files. The controller's third-party extractor is fetched lazily from the pinned upstream commit and hash-checked, rather than redistributed inline. This packaging adaptation is recorded in the export provenance. The scientific routing/cache code otherwise retains its source form. Historical hash guards describe the original sealed workspace and are not relaxed to make this source-only copy appear complete.
