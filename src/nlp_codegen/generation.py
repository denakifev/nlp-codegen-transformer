import time

import torch
from torch.nn.utils.rnn import pad_sequence


@torch.inference_mode()
def generate_codes(
    model,
    sources,
    tokenizer,
    device,
    pad_id,
    bos_id,
    eos_id,
    batch_size=64,
    max_new_tokens=512,
    decoding="greedy",
    temperature=0.7,
    top_p=0.9,
    num_beams=4,
    length_penalty=1.0,
):
    if batch_size < 1:
        raise ValueError("generation batch_size must be positive")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if model.training:
        raise ValueError("Call model.eval() before generation")
    device = torch.device(device)
    sources = [torch.as_tensor(source, dtype=torch.long, device="cpu") for source in sources]
    if any(source.ndim != 1 or not len(source) for source in sources):
        raise ValueError("Each generation source must be a non-empty sequence")
    order = sorted(range(len(sources)), key=lambda index: len(sources[index]))
    codes = [""] * len(sources)
    generated_tokens = 0
    unfinished_eos = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        source_ids = pad_sequence(
            [sources[index] for index in indices],
            batch_first=True,
            padding_value=pad_id,
        ).to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            generated = model.generate(
                source_ids, max_new_tokens=max_new_tokens, decoding=decoding,
                temperature=temperature, top_p=top_p, num_beams=num_beams,
                length_penalty=length_penalty,
            ).tolist()
        for index, ids in zip(indices, generated):
            if eos_id in ids:
                ids = ids[:ids.index(eos_id)]
            else:
                unfinished_eos += 1
            ids = [token for token in ids if token not in {pad_id, bos_id}]
            generated_tokens += len(ids)
            codes[index] = tokenizer.decode(ids, skip_special_tokens=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    seconds = time.perf_counter() - started
    return codes, {
        "generated_examples": len(sources),
        "generated_tokens": generated_tokens,
        "unfinished_eos": unfinished_eos,
        "generation_seconds": seconds,
        "generation_tokens_per_second": generated_tokens / max(seconds, 1e-9),
    }
