"""Pinned replay loader. No unpinned model/tokenizer fallback is permitted."""
PINS = {
    'mamba_130m': ('state-spaces/mamba-130m-hf', '1e76775f628fbf1350fbe4dbb3d971ba64af25a1'),
    'mamba_2_8b': ('state-spaces/mamba-2.8b-hf', '96c48e0292b63f5346b6d30061af2551f7101e26'),
    'zephyr_conversion': ('han1997/mamba-2.8b-zephyr-hf', '99a87b5e5f211c42595df370de50910f05dc7854'),
}


def load(name, require_cuda=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no model download attempted')
    model_id, revision = PINS[name]
    device = 'cuda' if require_cuda else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                               dtype=torch.float32, use_safetensors=True).to(device).eval()
    model.requires_grad_(False)
    tokenizer.clean_up_tokenization_spaces = False
    return model, tokenizer
