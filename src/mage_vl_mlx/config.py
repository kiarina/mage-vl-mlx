"""Configuration dataclasses mirroring the official Mage-VL config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VisionConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    patch_size: int = 16
    num_channels: int = 3
    out_hidden_size: int = 2560
    spatial_merge_size: int = 2
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    frame_windows_size: int = 4

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, data: dict) -> "VisionConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass
class TextConfig:
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    rope_theta: float = 5000000.0
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "TextConfig":
        fields = cls.__dataclass_fields__
        merged = {k: v for k, v in data.items() if k in fields}
        rope = data.get("rope_parameters") or {}
        if "rope_theta" in rope:
            merged["rope_theta"] = rope["rope_theta"]
        return cls(**merged)


@dataclass
class MageVLConfig:
    vision: VisionConfig
    text: TextConfig
    image_token_id: int = 151655

    @classmethod
    def from_json(cls, path: str | Path) -> "MageVLConfig":
        data = json.loads(Path(path).read_text())
        return cls(
            vision=VisionConfig.from_dict(data["vision_config"]),
            text=TextConfig.from_dict(data["text_config"]),
            image_token_id=data.get("image_token_id", 151655),
        )
