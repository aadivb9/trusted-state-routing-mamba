"""Recompute saved development scores without loading the model or evaluator."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions(task, response):
    # Independent transcription of the sealed strict rule and published parser.
    text = response.lower().lstrip()
    text = re.sub(r"^answer\s*:\s*", "", text).lstrip("\"' ")
    labels = {
        "sst2": [("negative", 0), ("positive", 1)],
        "sms_spam": [("not spam", 0), ("no spam", 0), ("spam", 1)],
        "mrpc": [("not equivalent", 0), ("equivalent", 1)],
    }[task]
    strict = None
    for label, value in labels:
        if text.startswith(label) and (len(text) == len(label) or
                                      not (text[len(label)].isalnum() or text[len(label)] == "_")):
            strict = value
            break
    text = response.lower().replace("answer: ", "").replace("answer:", "")
    if not response:
        official = -1
    elif task == "sst2":
        official = 1 if "positive" in text else 0 if "negative" in text else 1 if "neutral" in text else 2
    elif task == "sms_spam":
        official = 0 if any(x in text for x in ["not spam", "no spam", "t spam"]) else 1 if "spam" in text else 2
    else:
        official = 0 if "not equivalent" in text else 1 if "equivalent" in text else 2
    return strict, official


def main():
    seal = json.loads((ROOT / "competence_seal.json").read_text())
    for name, expected in seal["files"].items():
        assert digest(ROOT / name) == expected, name
    result_path = ROOT / "development_run_01/results.json"
    result = json.loads(result_path.read_text())
    source = json.loads((ROOT / "prepared/development.json").read_text())
    source = {(r["dataset"], r["source_id"]): r for r in source}
    assert len(source) == len(result["rows"]) == 72
    seen = set()
    for row in result["rows"]:
        key = row["dataset"], row["source_id"]
        assert key not in seen
        seen.add(key)
        assert row["gold"] == source[key]["label"]
        strict, official = predictions(row["dataset"], row["response"])
        assert (strict, official) == (row["strict_prediction"], row["official_prediction"])
        assert row["strict_correct"] == (strict == row["gold"])
        assert row["official_correct"] == (official == row["gold"])
        assert 0 < len(row["generated_ids"]) <= seal["max_new_tokens"]
    summary = {}
    for task in seal["tasks"]:
        rows = [r for r in result["rows"] if r["dataset"] == task]
        assert len(rows) == 24 and sum(r["gold"] for r in rows) == 12
        empty = result["instruction_only"][task]
        empty_pred = predictions(task, empty["response"])[0]
        assert empty_pred == empty["strict_prediction"]
        strict = sum(r["strict_correct"] for r in rows)
        valid = sum(r["strict_prediction"] is not None for r in rows)
        without = sum(r["gold"] == empty_pred for r in rows)
        gain = (strict - without) / len(rows)
        t = seal["thresholds"]
        summary[task] = dict(n=24, strict_correct=strict,
                             official_correct=sum(r["official_correct"] for r in rows),
                             strict_valid=valid, instruction_only_correct=without,
                             input_gain=gain,
                             passed=strict >= t["strict_correct_at_least"] and
                             valid >= t["strict_valid_at_least"] and gain >= t["input_gain_at_least"])
    assert summary == result["summary"]
    assert result["data_sha256"] == seal["files"]["prepared/development.json"]
    assert result["code_sha256"] == seal["files"]["run_competence.py"]
    assert result["protocol_sha256"] == seal["files"]["PROTOCOL.md"]
    hardware = json.loads((ROOT.parent / "mamba_28b_feasibility/fp32_run_02/results.json").read_text())
    assert hardware["backbone_before"] == hardware["backbone_after"] == result["backbone_before"] == result["backbone_after"]
    assert result["backbone_unchanged"]
    assert result["gate_passed"] == all(s["passed"] for s in summary.values())
    audit = dict(status="passed", rows_recomputed=72, summaries_recomputed=3,
                 sealed_files_verified=len(seal["files"]),
                 result_sha256=digest(result_path), gate_passed=result["gate_passed"],
                 limitation="Audits saved responses and recorded weight hashes; does not rerun GPU inference or independently decode token IDs.")
    (ROOT / "development_run_01/verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
