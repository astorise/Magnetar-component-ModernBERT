"""Execute pinned Transformers ModernBERT layers with synthetic weights.

Usage: python generate_encoder_reference.py path/to/modeling_modernbert.py
Requires PyTorch 2.15. The exact source definitions of the embeddings,
rotary embedding and encoder layers run with an eager attention interface.
Masks are constructed for one padded sequence and inclusive local windows.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

source = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(source).hexdigest() == "29b598c408582bd53c0a48ae34dce2cb008aefc5f6087b03c310b5ce7bb0de9b"
tree = ast.parse(source)
names = {"ModernBertEmbeddings", "ModernBertMLP", "ModernBertRotaryEmbedding", "ModernBertAttention", "ModernBertEncoderLayer", "eager_attention_forward", "rotate_half", "apply_rotary_pos_emb"}
nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]


class AttentionFunctions:
    def get_interface(self, name, default):
        assert name == "eager"
        return default


scope = {"torch": torch, "nn": nn, "ACT2FN": {"gelu": nn.GELU()},
         "GradientCheckpointingLayer": nn.Module,
         "dynamic_rope_update": lambda fn: fn,
         "use_kernel_forward_from_hub": lambda *args, **kwargs: lambda value: value,
         "use_kernelized_func": lambda *args, **kwargs: lambda value: value,
         "ALL_ATTENTION_FUNCTIONS": AttentionFunctions()}
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[])), "pinned_modernbert.py", "exec"), scope)


def patterned(name, shape):
    count = 1
    for dim in shape:
        count *= dim
    seed = sum((i + 1) * byte for i, byte in enumerate(name.encode("ascii"))) % 97
    values = ((torch.arange(count, dtype=torch.int64) * 17 + seed) % 101 - 50).float() / 200.0
    if "norm" in name and name.endswith(".weight"):
        values += 1.0
    return values.reshape(shape)


class Reference(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = scope["ModernBertEmbeddings"](cfg)
        self.layers = nn.ModuleList([scope["ModernBertEncoderLayer"](cfg, i) for i in range(len(cfg.layer_types))])
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.rotary = scope["ModernBertRotaryEmbedding"](cfg)

    def forward(self, ids, mask, positions, radius):
        hidden = self.embeddings(ids)
        length = ids.shape[1]
        for layer in self.layers:
            allowed = mask[:, None, None, :].expand(-1, 1, length, -1).bool().clone()
            if layer.attention_type == "sliding_attention":
                row = torch.arange(length)
                allowed &= ((row[:, None] - row[None, :]).abs() <= radius)[None, None]
            assert allowed.any(-1).all()
            additive = torch.zeros(allowed.shape, dtype=hidden.dtype).masked_fill(~allowed, torch.finfo(hidden.dtype).min)
            rotary = self.rotary(hidden, positions, layer.attention_type)
            hidden = layer(hidden, attention_mask=additive, position_embeddings=rotary)
        return self.final_norm(hidden)


results = []
for bias, ids, mask, positions in [
    (False, [1, 4, 6, 3, 9, 0, 0], [1, 1, 1, 1, 1, 0, 0], [0, 1, 2, 3, 4, 5, 6]),
    (True, [7, 5, 3, 2, 1], [1, 1, 1, 1, 1], [3, 4, 5, 6, 7]),
]:
    cfg = SimpleNamespace(vocab_size=24, hidden_size=8, intermediate_size=12, num_attention_heads=2,
        pad_token_id=0, norm_eps=1e-5, norm_bias=bias, embedding_dropout=0.0,
        mlp_bias=bias, mlp_dropout=0.0, hidden_activation="gelu", attention_bias=bias,
        attention_dropout=0.0, deterministic_flash_attn=False, sliding_window=2,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        max_position_embeddings=64, _attn_implementation="eager",
        rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 160000.0},
                         "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}})
    model = Reference(cfg).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(patterned(name, parameter.shape))
        output = model(torch.tensor([ids]), torch.tensor([mask]), torch.tensor([positions]), 2)
    results.append({"bias": bias, "ids": ids, "mask": mask, "positions": positions,
                    "output": output.flatten().tolist()})

Path(__file__).with_name("encoder_reference.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
