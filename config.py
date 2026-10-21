from dataclasses import dataclass, fields


@dataclass(frozen=True)  # Keep configuration values immutable.
class Config:
    vocab_size: int = 32000
    hidden_size: int = 768
    n_layers: int = 8
    n_heads: int = 12
    intermediate_size: int = 1792
    output_size: int = 768
    max_len: int = 256
    pad_id: int = 0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    init_std: float = 0.02
    seed: int = 42
    temperature: float = 0.05
    chunk_targets: tuple[int, ...] = (64, 156, 256)

    @classmethod
    def from_dict(cls, saved: dict) -> "Config":
        names = set()
        for field in fields(cls):
            names.add(field.name)
        config_values = {}
        for key, value in saved.items():
            if key in names:
                config_values[key] = value

        return cls(**config_values)
