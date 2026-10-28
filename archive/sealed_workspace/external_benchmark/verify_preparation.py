"""Check the prepared records against the pinned upstream selection function."""
import ast
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import prepare_data as prep
from upstream.OpenPromptInjection.utils.process_txt import open_txt


def upstream_select(rows):
    tree = ast.parse((prep.ROOT / "upstream/OpenPromptInjection/tasks/Task.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Task")
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == "__split_dataset_and_save")
    saved = {}
    namespace = {
        "defaultdict": defaultdict, "tqdm": lambda x: x,
        # Only array construction and file writing are substituted. The original
        # selection, balancing, traversal and sorting function executes unchanged.
        "np": SimpleNamespace(array=lambda value: value),
        "atomic_savez": lambda path, **data: saved.update({path.name: data}),
        "LOGGER": SimpleNamespace(info=lambda *args: None),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<pinned upstream selection>", "exec"), namespace)
    fake = SimpleNamespace(num=100, class_num=2, task_type="classification",
                           preprocess_func=lambda r: (r["data"], r["label"]),
                           get_data_saving_path=lambda: Path("/unused"))
    namespace[function.name](fake, rows)
    return saved


def main():
    sources, _ = prep.read_sources()
    checked = 0
    for name, rows in sources.items():
        expected = upstream_select(rows)
        actual = prep.official_selection(rows)
        for role, selected in zip(["target", "injected"], actual):
            reference = expected[role + "_data.npz"]
            assert reference["data"] == [r["data"] for r in selected], (name, role)
            assert reference["label"] == [r["label"] for r in selected], (name, role)
            checked += len(selected)
    folder = prep.ROOT / "prepared"
    audit = json.loads((folder / "audit.json").read_text())
    for file, expected in audit["files"].items():
        assert prep.digest(folder / file) == expected, file
    dev = json.loads((folder / "development.json").read_text())
    final = json.loads((folder / "clean.json").read_text()) + json.loads((folder / "injected_baselines.json").read_text())
    prompt_dir = prep.ROOT / "upstream/data/system_prompts"
    for r in dev + final[:300]:
        expected = open_txt(prompt_dir / (prep.TASKS[r["dataset"]] + ".txt"))[0]
        assert r["instruction"] == expected
    for r in final[300:]:
        expected = open_txt(prompt_dir / (prep.TASKS[r["dataset"]] + "_inject.txt"))[0]
        assert r["instruction"] == expected
    attacks = json.loads((folder / "attacks.json").read_text())
    selected = {name: prep.official_selection(rows) for name, rows in sources.items()}
    for row in attacks:
        target, injected = row["target_dataset"], row["injected_dataset"]
        index = int(row["id"].rsplit("/", 1)[1])
        instruction = open_txt(prompt_dir / (prep.TASKS[injected] + "_inject.txt"))[0]
        task = prep.PublishedTask(selected[injected][1], instruction)
        attacker = prep.ATTACKS[row["attack"]](row["attack"], task)
        expected = attacker.inject(prep.format_text(selected[target][0][index]["data"]), index, prep.TASKS[target])
        assert row["untrusted_text"] == expected, row["id"]
    assert not {prep.identity(r["text"]) for r in dev} & {prep.identity(r["text"]) for r in final}
    # MRPC source sentences may be reused across different pairs: inspect this too.
    dev_sentences = {prep.identity(s) for r in dev if r["dataset"] == "mrpc" for s in r["data"]}
    final_sentences = {prep.identity(s) for r in final if r["dataset"] == "mrpc" for s in r["data"]}
    result = {"status": "passed", "official_selection_records_verified": checked,
              "official_attack_strings_verified": len(attacks),
              "official_instruction_strings_verified": len(dev) + len(final),
              "prepared_file_hashes_verified": len(audit["files"]),
              "development_final_text_overlap": 0,
              "mrpc_development_final_shared_sentences": len(dev_sentences & final_sentences),
              "independent_final_source_texts": len({prep.identity(r["text"]) for r in final}),
              "scope": "Data-selection parity; not model performance or complete upstream runtime parity"}
    (prep.ROOT / "preparation_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
