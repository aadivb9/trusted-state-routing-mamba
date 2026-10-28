"""Final descriptive clean/injected baseline evaluation; no attack efficacy claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from external_benchmark.run_competence import MODEL, REVISION, official_parsers, strict_label


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run(output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import dual_state_mamba as core

    output.mkdir(parents=True, exist_ok=False)
    seal_path = ROOT / 'research_closeout/seal.json'
    seal = json.loads(seal_path.read_text())
    for name, expected in seal['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    path = output / 'results.json'
    result = dict(status='running', evaluation='post_development_final_baseline_diagnostic',
                  attack_generations_evaluated=0, model=MODEL, revision=REVISION,
                  dtype='float32', max_new_tokens=32, batch_size=1,
                  seal_sha256=hashlib.sha256(seal_path.read_bytes()).hexdigest(),
                  versions={'torch':torch.__version__, 'transformers':transformers.__version__},
                  rows=[], instruction_only={}, summary={})
    save(path, result)
    try:
        assert torch.cuda.is_available()
        result['gpu'] = torch.cuda.get_device_name(0)
        assert 'T4' in result['gpu']
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.float32, device_map={'':0},
            low_cpu_mem_usage=True, use_safetensors=True).eval()
        model.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        tokenizer.clean_up_tokenization_spaces = False
        result['backbone_before'] = core.model_fingerprint(model)
        parsers = official_parsers()

        def generate(prompt):
            ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids'].to(model.device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = model.generate(input_ids=ids, max_new_tokens=32, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            tokens = generated[0, ids.shape[1]:].tolist()
            return dict(response=tokenizer.decode(tokens, skip_special_tokens=True), generated_ids=tokens,
                        input_tokens=ids.shape[1], elapsed_seconds=time.perf_counter()-start)

        with torch.inference_mode():
            for role, name in [('target','clean.json'), ('injected','injected_baselines.json')]:
                data = json.loads((ROOT / 'external_benchmark/prepared' / name).read_text())
                for task in ['sst2','sms_spam','mrpc']:
                    examples = [r for r in data if r['dataset'] == task]
                    assert len(examples) == 100 and sum(r['label'] for r in examples) == 50
                    instruction = examples[0]['instruction']
                    control_key = hashlib.sha256(instruction.encode()).hexdigest()
                    if control_key not in result['instruction_only']:
                        result['instruction_only'][control_key] = dict(instruction=instruction, **generate(instruction+'\nText: '))
                    for record in examples:
                        row = dict(role=role, dataset=task, source_id=record['source_id'], gold=record['label'],
                                   instruction_control=control_key,
                                   **generate(record['instruction']+'\nText: '+record['text']))
                        row['strict_prediction'] = strict_label(task, row['response'])
                        row['official_prediction'] = parsers[task](row['response'])
                        row['strict_correct'] = row['strict_prediction'] == row['gold']
                        row['official_correct'] = row['official_prediction'] == row['gold']
                        result['rows'].append(row)
                        save(path, result)
                        if len(result['rows']) % 10 == 0:
                            print('PROGRESS',len(result['rows']),'/600', flush=True)
                    measured = [r for r in result['rows'] if r['role']==role and r['dataset']==task]
                    result['summary'][role+'/'+task] = dict(n=len(measured),
                        strict_correct=sum(r['strict_correct'] for r in measured),
                        official_correct=sum(r['official_correct'] for r in measured),
                        strict_valid=sum(r['strict_prediction'] is not None for r in measured))
                    save(path, result)
            result['backbone_after'] = core.model_fingerprint(model)
            result['backbone_unchanged'] = result['backbone_before'] == result['backbone_after']
            assert result['backbone_unchanged']
        result['status'] = 'complete'
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save(path,result)
        raise
    save(path,result)
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
