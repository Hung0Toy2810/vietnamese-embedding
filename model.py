import math
from typing import cast
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from config import Config


# Normalize each token vector using its root mean square.
class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        original_dtype = x.dtype
        x = x.float()

        mean_square = x.square().mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        normalized = x / rms

        scaled = normalized * self.weight.float()
        return scaled.to(original_dtype)


def apply_rope(x: Tensor, theta: float) -> Tensor:
    dtype, length, dim = x.dtype, x.shape[-2], x.shape[-1]
    # Rotation stays in fp32 even when the encoder runs under autocast.
    with torch.autocast(device_type=x.device.type, enabled=False):
        inv = theta ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
        angle = torch.arange(length, device=x.device).float()[:, None] * inv[None, :]
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        rotated = torch.stack(
            (
                even * angle.cos() - odd * angle.sin(),
                even * angle.sin() + odd * angle.cos(),
            ),
            dim=-1,
        )
    return rotated.flatten(-2).to(dtype)


class Attention(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.heads, self.dim, self.theta = (
            c.n_heads,
            c.hidden_size // c.n_heads,
            c.rope_theta,
        )
        self.qkv = nn.Linear(c.hidden_size, 3 * c.hidden_size, bias=False)
        self.out = nn.Linear(c.hidden_size, c.hidden_size, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(q, self.theta), apply_rope(k, self.theta)
        # SDPA boolean True means a key is visible, unlike MultiheadAttention.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:, None, None, :], dropout_p=0.0, is_causal=False
        )
        return self.out(out.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, c: Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(c.hidden_size, c.norm_eps)
        self.ffn_norm = RMSNorm(c.hidden_size, c.norm_eps)
        self.attn = Attention(c)
        self.gate = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        y = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class Student(nn.Module):
    def __init__(self, c: Config = Config()) -> None:
        super().__init__()
        if c.hidden_size % c.n_heads or (c.hidden_size // c.n_heads) % 2:
            raise ValueError("RoPE requires an even integer head dimension")
        self.config: Config = c
        self.embedding = nn.Embedding(c.vocab_size, c.hidden_size, padding_idx=c.pad_id)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.norm = RMSNorm(c.hidden_size, c.norm_eps)
        self.projection = nn.Linear(c.hidden_size, c.output_size, bias=False)
        self.apply(self._initialize)
        for layer in self.blocks:
            block = cast(Block, layer)
            for weight in (block.attn.out.weight, block.down.weight):
                nn.init.normal_(weight, std=c.init_std / math.sqrt(2 * c.n_layers))
        with torch.no_grad():
            self.embedding.weight[c.pad_id].zero_()

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, std=self.config.init_std)

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        if ids.shape[1] > self.config.max_len:
            raise ValueError("Sequence exceeds max_len")
        x = self.embedding(ids)
        for layer in self.blocks:
            block = cast(Block, layer)
            x = block(x, mask.bool())
        x = self.norm(x).float()
        weights = mask.unsqueeze(-1).float()
        pooled = (x * weights).sum(1) / weights.sum(1).clamp_min(1)
        return F.normalize(self.projection(pooled).float(), p=2, dim=-1)

    def count_params(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "non_embedding": total - self.embedding.weight.numel()}


if __name__ == "__main__":
    print(Student().count_params())
