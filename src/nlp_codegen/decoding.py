import math

import torch


def validate_decoding(decoding, temperature, top_p, num_beams, length_penalty):
    if decoding not in {"greedy", "top_p", "beam"}:
        raise ValueError("decoding must be greedy, top_p or beam")
    if decoding == "top_p":
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
    if decoding == "beam":
        if type(num_beams) is not int or num_beams < 1:
            raise ValueError("num_beams must be a positive integer")
        if not math.isfinite(length_penalty) or length_penalty < 0:
            raise ValueError("length_penalty must be finite and nonnegative")


def sample_top_p(logits, temperature, top_p):
    logits = logits.float()
    probabilities = ((logits - logits.amax(dim=-1, keepdim=True)) / temperature).softmax(dim=-1)
    if top_p == 1:
        return torch.multinomial(probabilities, 1).squeeze(-1)
    probabilities, indices = probabilities.sort(dim=-1, descending=True)
    remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
    probabilities = probabilities.masked_fill(remove, 0)
    sampled = torch.multinomial(probabilities, 1)
    return indices.gather(-1, sampled).squeeze(-1)


def reorder_beam_cache(cache, parents):
    length = cache["position"]
    for layer in cache["layers"]:
        for name in ("self_key", "self_value"):
            prefix = layer[name][:, :, :length]
            prefix.copy_(prefix.index_select(0, parents))
    prefix = cache["target_mask"][:, :length]
    prefix.copy_(prefix.index_select(0, parents))



@torch.inference_mode()
def beam_search(model, source_ids, max_new_tokens, num_beams, length_penalty):
    batch_size = source_ids.size(0)
    device = source_ids.device
    memory, source_mask = model.encode(source_ids)
    memory = memory.repeat_interleave(num_beams, dim=0)
    source_mask = source_mask.repeat_interleave(num_beams, dim=0)
    cache = model.init_cache(memory, source_mask, max_new_tokens)
    token_ids = source_ids.new_full((batch_size * num_beams, 1), model.bos_id)
    sequences = source_ids.new_full((batch_size, num_beams, max_new_tokens), model.pad_id)
    scores = torch.full((batch_size, num_beams), -torch.inf, device=device)
    scores[:, 0] = 0
    finished_sequences = sequences.clone()
    finished_scores = scores.new_full(scores.shape, -torch.inf)
    finished_lengths = source_ids.new_zeros((batch_size, num_beams))
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    offsets = torch.arange(batch_size, device=device)[:, None] * num_beams

    for position in range(max_new_tokens):
        logits, cache = model.decode_step(token_ids, cache)
        logits = logits[:, 0].float()
        logits[:, model.pad_id] = -torch.inf
        logits[:, model.bos_id] = -torch.inf
        vocab_size = logits.size(-1)
        candidates = logits.log_softmax(dim=-1).view(batch_size, num_beams, vocab_size)
        candidates = candidates + scores[:, :, None]
        candidate_scores, candidate_ids = candidates.flatten(1).topk(2 * num_beams, dim=-1)
        parents = candidate_ids // vocab_size
        tokens = candidate_ids % vocab_size
        candidate_sequences = sequences.gather(
            1, parents[:, :, None].expand(-1, -1, max_new_tokens)
        )
        candidate_sequences[:, :, position] = tokens
        rank = torch.arange(2 * num_beams, device=device)[None]
        terminal = tokens.eq(model.eos_id) & (rank < num_beams) & ~done[:, None]
        terminal_scores = (candidate_scores / (position + 1) ** length_penalty).masked_fill(~terminal, -torch.inf)

        all_scores = torch.cat((finished_scores, terminal_scores), dim=1)
        all_sequences = torch.cat((finished_sequences, candidate_sequences), dim=1)
        all_lengths = torch.cat((
            finished_lengths,
            source_ids.new_full(candidate_scores.shape, position + 1),
        ), dim=1)
        finished_scores, best_finished = all_scores.topk(num_beams, dim=-1)
        finished_sequences = all_sequences.gather(
            1, best_finished[:, :, None].expand(-1, -1, max_new_tokens)
        )
        finished_lengths = all_lengths.gather(1, best_finished)

        active_scores = candidate_scores.masked_fill(tokens.eq(model.eos_id), -torch.inf)
        scores, best_active = active_scores.topk(num_beams, dim=-1)
        parents = parents.gather(1, best_active)
        tokens = tokens.gather(1, best_active)
        sequences = candidate_sequences.gather(
            1, best_active[:, :, None].expand(-1, -1, max_new_tokens)
        )
        reorder_beam_cache(cache, (parents + offsets).flatten())
        enough_finished = torch.isfinite(finished_scores).all(dim=-1)
        best_active_score = scores[:, 0] / (position + 1) ** length_penalty
        done = done | (enough_finished & (finished_scores[:, -1] >= best_active_score))
        done = done | ~torch.isfinite(scores).any(dim=-1)
        scores = scores.masked_fill(done[:, None], -torch.inf)
        token_ids = tokens.masked_fill(done[:, None], model.pad_id).reshape(-1, 1)
        if done.all():
            break

    active_score = scores[:, 0] / (position + 1) ** length_penalty
    use_active = active_score > finished_scores[:, 0]
    result = torch.where(use_active[:, None], sequences[:, 0], finished_sequences[:, 0])
    lengths = torch.where(use_active, source_ids.new_full((batch_size,), position + 1), finished_lengths[:, 0])
    return result[:, :int(lengths.max().item())]
