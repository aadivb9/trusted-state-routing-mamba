"""Prepare a pinned three-task Open-Prompt-Injection subset from original data.

No LLM generation, model inference, or test-driven prompt selection occurs here.
Published constructors build every attack; a separate development set gates
model competence before any final benchmark answers are evaluated.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from upstream.OpenPromptInjection.attackers.NaiveAttacker import NaiveAttacker
from upstream.OpenPromptInjection.attackers.EscapeCharAttacker import EscapeCharAttacker
from upstream.OpenPromptInjection.attackers.IgnoreAttacker import IgnoreAttacker
from upstream.OpenPromptInjection.attackers.FakeCompAttacker import FakeCompAttacker
from upstream.OpenPromptInjection.attackers.CombineAttacker import CombineAttacker

ROOT = Path(__file__).resolve().parent
TASKS = {"sst2": "sentiment_analysis", "sms_spam": "spam_detection",
         "mrpc": "duplicate_sentence_detection"}
ATTACKS = {"naive": NaiveAttacker, "escape": EscapeCharAttacker,
           "ignore": IgnoreAttacker, "fake_completion": FakeCompAttacker,
           "combine": CombineAttacker}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(text):
    return " ".join(str(text).lower().split())


def format_text(value):
    if isinstance(value, list):
        a, b = [s.replace("\n", "") for s in value]
        return f"Sentence1: '{a}', Sentence2: '{b}'"
    return value


def balanced(rows, n):
    counts = {0: 0, 1: 0}
    selected = []
    for row in rows:
        if counts[row["label"]] < n // 2:
            selected.append(row)
            counts[row["label"]] += 1
        if len(selected) == n:
            break
    if len(selected) != n:
        raise ValueError("Insufficient balanced source records")
    return selected


def official_selection(rows, n=100):
    # Task.__split_dataset_and_save fills targets, then injections in source order,
    # and sorts (label, data) descending for targets and ascending for injections.
    target = balanced(rows, n)
    last = rows.index(target[-1])
    injected = balanced(rows[last + 1:], n)
    key = lambda r: (r["label"], r["data"])
    return sorted(target, key=key, reverse=True), sorted(injected, key=key)


def read_sources():
    raw = ROOT / "raw"
    with zipfile.ZipFile(raw / "sst2.zip") as archive:
        sst = {}
        for split, file in [("train", "train.tsv"), ("validation", "dev.tsv")]:
            rows = csv.DictReader(io.StringIO(archive.read("SST-2/" + file).decode()),
                                  delimiter="\t", quoting=csv.QUOTE_NONE)
            sst[split] = [{"source_id": f"sst2/{split}/{i}", "data": r["sentence"],
                           "label": int(r["label"])} for i, r in enumerate(rows)]
    with zipfile.ZipFile(raw / "sms_spam.zip") as archive:
        sms = []
        for i, line in enumerate(archive.read("SMSSpamCollection").decode().splitlines(keepends=True)):
            fields = line.split("\t")
            sms.append({"source_id": f"sms_spam/train/{i}", "data": fields[1],
                        "label": int(fields[0] != "ham")})
    mrpc = {}
    dev_ids = {tuple(line.split("\t")) for line in
               (raw / "mrpc_dev_ids.tsv").read_text().splitlines()}
    for split in ["train", "test"]:
        with (raw / f"mrpc_{split}.tsv").open(encoding="utf-8-sig") as stream:
            rows = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
            mrpc[split] = [{"source_id": f"mrpc/{split}/{r['#1 ID']}/{r['#2 ID']}",
                            "data": [r["#1 String"], r["#2 String"]],
                            "label": int(r["Quality"])} for r in rows
                           if split == "test" or (r["#1 ID"], r["#2 ID"]) not in dev_ids]
    # The official final data sources are SST2 validation, SMS train, MRPC test.
    return {"sst2": sst["validation"], "sms_spam": sms, "mrpc": mrpc["test"]}, {
        "sst2": sst["train"], "sms_spam": sms[-200:], "mrpc": mrpc["train"]}


class PublishedTask:
    def __init__(self, records, instruction):
        self.records, self.instruction = records, instruction

    def get_instruction(self):
        return self.instruction

    def __getitem__(self, index):
        row = self.records[index]
        return format_text(row["data"]), row["label"]


def main():
    output = ROOT / "prepared"
    output.mkdir(exist_ok=False)
    manifest = json.loads((ROOT / "upstream_manifest.json").read_text())
    for file, expected in manifest["files"].items():
        assert digest(ROOT / "upstream" / file) == expected, file
    for row in json.loads((ROOT / "raw/download_manifest.json").read_text()):
        assert digest(ROOT / "raw" / row["file"]) == row["sha256"]
    sources, dev_sources = read_sources()
    selected, instructions, injected_instructions = {}, {}, {}
    for name, task in TASKS.items():
        selected[name] = official_selection(sources[name])
        folder = ROOT / "upstream/data/system_prompts"
        # Match upstream utils.open_txt: trailing whitespace is removed. This
        # also controls whether attack constructors replace the final period.
        instructions[name] = (folder / (task + ".txt")).read_text().splitlines()[0].rstrip()
        injected_instructions[name] = (folder / (task + "_inject.txt")).read_text().splitlines()[0].rstrip()
    development, clean, injected_baselines, attacks = [], [], [], []
    final_content = {identity(format_text(r["data"])) for pair in selected.values()
                     for role in pair for r in role}
    for name in TASKS:
        used = set()
        candidates = []
        for r in dev_sources[name]:
            text_id = identity(format_text(r["data"]))
            if text_id not in final_content and text_id not in used:
                candidates.append(r)
                used.add(text_id)
        for r in balanced(candidates, 24):
            development.append({**r, "dataset": name, "instruction": instructions[name],
                                "text": format_text(r["data"])})
        for role, rows in zip(["target", "injected"], selected[name]):
            destination = clean if role == "target" else injected_baselines
            for r in rows:
                destination.append({**r, "dataset": name,
                                    "instruction": instructions[name] if role == "target" else injected_instructions[name],
                                    "text": format_text(r["data"])})
    for target_name, task_name in TASKS.items():
        for injection_name in TASKS:
            task = PublishedTask(selected[injection_name][1], injected_instructions[injection_name])
            for attack_name, cls in ATTACKS.items():
                attacker = cls(attack_name, task)
                for i, row in enumerate(selected[target_name][0]):
                    text = attacker.inject(format_text(row["data"]), i, task_name)
                    attacks.append({"id": f"{target_name}/{injection_name}/{attack_name}/{i:03}",
                                    "target_dataset": target_name, "injected_dataset": injection_name,
                                    "attack": attack_name, "target_source_id": row["source_id"],
                                    "injected_source_id": selected[injection_name][1][i]["source_id"],
                                    "target_label": row["label"], "injected_label": task[i][1],
                                    "instruction": instructions[target_name], "untrusted_text": text})
    assert len(development) == 72 and len(clean) == 300 and len(attacks) == 4500
    assert not final_content & {identity(r["text"]) for r in development}
    for name, rows in [("development", development), ("clean", clean),
                       ("injected_baselines", injected_baselines), ("attacks", attacks)]:
        (output / (name + ".json")).write_text(json.dumps(rows, indent=2) + "\n")
    audit = {"status": "prepared_not_evaluated", "upstream_commit": manifest["commit"],
             "counts": {"development": 72, "clean": 300, "injected_baselines": 300, "attacked": 4500},
             "development_final_normalized_text_overlap": 0,
             "attack_generation": "Unmodified official attack classes imported from pinned source",
             "selection": "Equivalent stdlib port of official first-balanced-100 selection; independent parity check required",
             "scope": "Three-task subset of Open-Prompt-Injection, not the full benchmark",
             "files": {p.name: digest(p) for p in output.glob("*.json")}}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
