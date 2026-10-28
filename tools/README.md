# Public checks

Run `python3 tools/verify_public.py` and `python3 tools/run_public_checks.py` from the repository root. They verify exported bytes, included saved-response/timing records and the data-independent implementation controls. They do not perform inference or recreate withheld source datasets.

Other scripts are retained for code review and describe the full local evidence workspace. In particular, `verify.py`, package builders and full-suite historical fault-injection tests require files that are deliberately absent from the public export. They fail closed; their criteria are not weakened here.
