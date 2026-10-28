"""CUDA-only streamed loading for the prospective large-model run."""
from src.runtime.pinned_models import PINS


def load(name):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no checkpoint download attempted')
    model_id,revision=PINS[name]
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision)
    model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,dtype=torch.float32,
                                             use_safetensors=True,device_map={'':0},low_cpu_mem_usage=True).eval()
    model.requires_grad_(False);tokenizer.clean_up_tokenization_spaces=False
    return model,tokenizer
