"""Top-level Mage-VL model: vision tower + Qwen3 decoder."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import MageVLConfig
from .language import KVCache, LanguageModel
from .vision import VisionModel


class MageVL(nn.Module):
    def __init__(self, config: MageVLConfig):
        super().__init__()
        self.config = config
        self.vision = VisionModel(config.vision)
        self.language = LanguageModel(config.text)
        self.lm_head = nn.Linear(config.text.hidden_size, config.text.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "MageVL":
        path = Path(path)
        model = cls(MageVLConfig.from_json(path / "config.json"))
        model.load_weights(str(path / "model.safetensors"))
        mx.eval(model.parameters())
        return model

    def embed(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        grid_thw: mx.array | None = None,
        patch_positions: mx.array | None = None,
    ) -> mx.array:
        """Embed tokens and splice vision features into the image placeholders."""
        embeds = self.language.embed_tokens(input_ids)
        if pixel_values is None:
            return embeds

        image_embeds = self.vision(pixel_values, grid_thw, patch_positions)
        image_embeds = image_embeds.astype(embeds.dtype)

        mask = input_ids == self.config.image_token_id
        positions = mx.array(
            [i for i, flag in enumerate(mask[0].tolist()) if flag], dtype=mx.int32
        )
        if positions.size != image_embeds.shape[0]:
            raise ValueError(
                f"image token count {positions.size} != vision features {image_embeds.shape[0]}"
            )
        embeds[0, positions] = image_embeds
        return embeds

    def __call__(self, embeds: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.lm_head(self.language(embeds, cache))

    def embed_video(
        self, input_ids: mx.array, video_path: str, **preprocess_kwargs
    ) -> mx.array:
        """Preprocess a video torch-free and embed it into the prompt."""
        from .video import preprocess_video

        processed = preprocess_video(video_path, **preprocess_kwargs)
        return self.embed(
            input_ids,
            mx.array(processed["pixel_values"]),
            mx.array(processed["grid_thw"].astype("int32")),
            mx.array(processed["patch_positions"].astype("int32")),
        )

    def generate(
        self,
        input_ids: mx.array,
        pixel_values: mx.array | None = None,
        grid_thw: mx.array | None = None,
        patch_positions: mx.array | None = None,
        max_new_tokens: int = 64,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Greedy decoding. Returns the generated token ids."""
        cache = [KVCache() for _ in self.language.layers]
        embeds = self.embed(input_ids, pixel_values, grid_thw, patch_positions)

        logits = self(embeds, cache)[:, -1]
        tokens: list[int] = []
        for _ in range(max_new_tokens):
            token = int(mx.argmax(logits, axis=-1).item())
            tokens.append(token)
            if eos_token_id is not None and token == eos_token_id:
                break
            next_embed = self.language.embed_tokens(mx.array([[token]]))
            logits = self(next_embed, cache)[:, -1]
        return tokens
