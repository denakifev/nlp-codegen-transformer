import math

import torch
from torch import nn
from torch.nn import functional as F

from nlp_codegen.decoding import beam_search, sample_top_p, validate_decoding


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(x.dtype)


def make_norm(width, norm_type="rmsnorm", eps=1e-6):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("norm_eps must be finite and positive")
    if norm_type == "rmsnorm":
        return RMSNorm(width, eps=eps)
    if norm_type == "layernorm":
        return nn.LayerNorm(width, eps=eps, elementwise_affine=True, bias=True)
    raise ValueError("norm_type must be rmsnorm or layernorm")


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.register_buffer("cosine", torch.empty(0), persistent=False)
        self.register_buffer("sine", torch.empty(0), persistent=False)

    def forward(self, length, device, dtype, offset=0):
        end = offset + length
        if (
            self.cosine.size(0) < end
            or self.cosine.device != device
            or self.cosine.dtype != torch.float32
            or self.cosine.is_inference()
            or self.sine.is_inference()
        ):
            cache_length = max(end, 128, self.cosine.size(0) * 2)
            with torch.inference_mode(False):
                frequency = self.theta ** (
                    -torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                    / self.head_dim
                )
                positions = torch.arange(cache_length, device=device, dtype=torch.float32)
                angles = positions[:, None] * frequency[None, :]
                self.cosine = angles.cos()
                self.sine = angles.sin()
        cosine = self.cosine[offset:end].to(dtype)[None, None]
        sine = self.sine[offset:end].to(dtype)[None, None]
        return cosine, sine


def apply_rotary(x, rotary):
    cosine, sine = rotary
    cosine = cosine.to(x.dtype)
    sine = sine.to(x.dtype)
    first, second = x.chunk(2, dim=-1)
    return torch.cat(
        (first * cosine - second * sine, second * cosine + first * sine), dim=-1
    )


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def project_query(self, x):
        batch_size, length = x.shape[:2]
        return self.query(x).view(
            batch_size, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def project_key_value(self, x):
        batch_size, length = x.shape[:2]
        shape = (batch_size, length, self.num_kv_heads, self.head_dim)
        key = self.key(x).view(shape).transpose(1, 2)
        value = self.value(x).view(shape).transpose(1, 2)
        return key, value

    def attend(self, query, key, value, mask):
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        result = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        batch_size, _, length, _ = result.shape
        result = result.transpose(1, 2).contiguous().view(batch_size, length, -1)
        return self.output(result)

    def forward(self, x, memory, mask, rotary=None):
        query = self.project_query(x)
        key, value = self.project_key_value(memory)
        if rotary is not None:
            query = apply_rotary(query, rotary)
            key = apply_rotary(key, rotary)
        return self.attend(query, key, value, mask)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, norm_type="rmsnorm", norm_eps=1e-6):
        super().__init__()
        self.attention_norm = make_norm(d_model, norm_type, norm_eps)
        self.attention = Attention(d_model, num_heads, num_heads, dropout)
        self.ffn_norm = make_norm(d_model, norm_type, norm_eps)
        self.ffn = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, rotary):
        normalized = self.attention_norm(x)
        x = x + self.dropout(self.attention(normalized, normalized, mask, rotary))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff, dropout, norm_type="rmsnorm", norm_eps=1e-6):
        super().__init__()
        self.self_norm = make_norm(d_model, norm_type, norm_eps)
        self.self_attention = Attention(d_model, num_heads, num_kv_heads, dropout)
        self.cross_norm = make_norm(d_model, norm_type, norm_eps)
        self.cross_attention = Attention(d_model, num_heads, num_kv_heads, dropout)
        self.ffn_norm = make_norm(d_model, norm_type, norm_eps)
        self.ffn = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, self_mask, source_mask, rotary):
        normalized = self.self_norm(x)
        x = x + self.dropout(
            self.self_attention(normalized, normalized, self_mask, rotary)
        )
        x = x + self.dropout(
            self.cross_attention(self.cross_norm(x), memory, source_mask)
        )
        return x + self.dropout(self.ffn(self.ffn_norm(x)))

    def decode_step(self, x, cache, position, self_mask, source_mask, rotary):
        normalized = self.self_norm(x)
        query = apply_rotary(self.self_attention.project_query(normalized), rotary)
        key, value = self.self_attention.project_key_value(normalized)
        key = apply_rotary(key, rotary)
        cache["self_key"][:, :, position : position + 1] = key
        cache["self_value"][:, :, position : position + 1] = value
        x = x + self.self_attention.attend(
            query,
            cache["self_key"][:, :, : position + 1],
            cache["self_value"][:, :, : position + 1],
            self_mask,
        )
        query = self.cross_attention.project_query(self.cross_norm(x))
        x = x + self.cross_attention.attend(
            query, cache["cross_key"], cache["cross_value"], source_mask
        )
        return x + self.ffn(self.ffn_norm(x))


class CodeTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = int(config["d_model"])
        num_heads = int(config["num_heads"])
        num_kv_heads = int(config.get("num_kv_heads", num_heads))
        d_ff = int(config["d_ff"])
        dropout = float(config.get("dropout", 0.0))
        norm_type = config.get("norm_type", "rmsnorm")
        norm_eps = float(config.get("norm_eps", 1e-6))
        if d_model % num_heads != 0 or (d_model // num_heads) % 2 != 0:
            raise ValueError("d_model / num_heads must be an even integer")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_kv_heads must divide num_heads")
        self.pad_id = int(config["pad_id"])
        self.bos_id = int(config["bos_id"])
        self.eos_id = int(config["eos_id"])
        self.embedding = nn.Embedding(
            int(config["vocab_size"]), d_model, padding_idx=self.pad_id
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(
            d_model // num_heads, float(config.get("rope_theta", 10000.0))
        )
        self.encoder = nn.ModuleList(
            EncoderLayer(d_model, num_heads, d_ff, dropout, norm_type, norm_eps)
            for _ in range(int(config["encoder_layers"]))
        )
        self.decoder = nn.ModuleList(
            DecoderLayer(d_model, num_heads, num_kv_heads, d_ff, dropout, norm_type, norm_eps)
            for _ in range(int(config["decoder_layers"]))
        )
        self.encoder_norm = make_norm(d_model, norm_type, norm_eps)
        self.decoder_norm = make_norm(d_model, norm_type, norm_eps)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def encode(self, source_ids):
        source_mask = source_ids.ne(self.pad_id)[:, None, None, :]
        x = self.embedding_dropout(self.embedding(source_ids))
        rotary = self.rotary(x.size(1), x.device, x.dtype)
        for layer in self.encoder:
            x = layer(x, source_mask, rotary)
        return self.encoder_norm(x), source_mask

    def forward(self, source_ids, target_ids):
        memory, source_mask = self.encode(source_ids)
        x = self.embedding_dropout(self.embedding(target_ids))
        length = target_ids.size(1)
        causal_mask = torch.ones(
            length, length, device=target_ids.device, dtype=torch.bool
        ).tril()
        self_mask = causal_mask[None, None] & target_ids.ne(self.pad_id)[
            :, None, None, :
        ]
        rotary = self.rotary(length, x.device, x.dtype)
        for layer in self.decoder:
            x = layer(x, memory, self_mask, source_mask, rotary)
        return F.linear(self.decoder_norm(x), self.embedding.weight)

    def init_cache(self, memory, source_mask, max_length):
        layers = []
        for layer in self.decoder:
            cross_key, cross_value = layer.cross_attention.project_key_value(memory)
            shape = (
                memory.size(0),
                layer.self_attention.num_kv_heads,
                max_length,
                layer.self_attention.head_dim,
            )
            layers.append(
                {
                    "self_key": cross_key.new_empty(shape),
                    "self_value": cross_value.new_empty(shape),
                    "cross_key": cross_key,
                    "cross_value": cross_value,
                }
            )
        return {
            "position": 0,
            "layers": layers,
            "source_mask": source_mask,
            "target_mask": torch.zeros(
                memory.size(0), max_length, device=memory.device, dtype=torch.bool
            ),
        }

    def decode_step(self, token_ids, cache):
        position = cache["position"]
        if token_ids.size(1) != 1:
            raise ValueError("decode_step expects one token per sequence")
        if position >= cache["target_mask"].size(1):
            raise ValueError("Decoder cache is full")
        cache["target_mask"][:, position] = token_ids[:, 0].ne(self.pad_id)
        self_mask = cache["target_mask"][:, None, None, : position + 1]
        x = self.embedding(token_ids)
        rotary = self.rotary(1, x.device, x.dtype, offset=position)
        for layer, layer_cache in zip(self.decoder, cache["layers"]):
            x = layer.decode_step(
                x,
                layer_cache,
                position,
                self_mask,
                cache["source_mask"],
                rotary,
            )
        cache["position"] = position + 1
        logits = F.linear(self.decoder_norm(x), self.embedding.weight)
        return logits, cache

    @torch.inference_mode()
    def generate(
        self, source_ids, max_new_tokens=256, decoding="greedy", temperature=0.7,
        top_p=0.9, num_beams=4, length_penalty=1.0,
    ):
        if self.training:
            raise ValueError("Call model.eval() before generation")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        validate_decoding(decoding, temperature, top_p, num_beams, length_penalty)
        if max_new_tokens == 0:
            return source_ids.new_empty((source_ids.size(0), 0))
        if decoding == "beam" and num_beams > 1:
            return beam_search(self, source_ids, max_new_tokens, num_beams, length_penalty)
        memory, source_mask = self.encode(source_ids)
        cache = self.init_cache(memory, source_mask, max_new_tokens)
        token_ids = source_ids.new_full((source_ids.size(0), 1), self.bos_id)
        generated = source_ids.new_full(
            (source_ids.size(0), max_new_tokens), self.pad_id
        )
        finished = torch.zeros(
            source_ids.size(0), device=source_ids.device, dtype=torch.bool
        )
        for position in range(max_new_tokens):
            logits, cache = self.decode_step(token_ids, cache)
            logits = logits[:, 0]
            logits[:, self.pad_id] = -torch.inf
            logits[:, self.bos_id] = -torch.inf
            next_ids = sample_top_p(logits, temperature, top_p) if decoding == "top_p" else logits.argmax(dim=-1)
            next_ids = next_ids.masked_fill(finished, self.pad_id)
            generated[:, position] = next_ids
            finished = finished | next_ids.eq(self.eos_id)
            token_ids = next_ids[:, None]
            if finished.all():
                break
        return generated[:, : position + 1]
