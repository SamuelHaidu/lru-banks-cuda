"""Simple GPT: a standard decoder-only Transformer baseline.

Not this project's usual spherical/normalized design -- a plain,
off-the-shelf GPT-2-style Transformer, built almost entirely from
`keras`/`keras_hub` ready-made layers (`keras.layers.Embedding`,
`keras_hub.layers.PositionEmbedding`, `keras_hub.layers.TransformerDecoder`,
which itself wraps KerasHub's fused `CachedMultiHeadAttention`). No custom
attention, no custom normalization tricks, no bias-free constraint, no
custom `train_step`. Its only purpose is to be a normal, well-understood
baseline to compare this project's own architectures against.

Single-target output: `call()` returns logits for only the last context
position, `(batch, vocabulary_size)`, matching `BaseLM`'s contract and
`model/data.py`'s single-next-token sampling -- exactly what every other
architecture in this project already does, so nothing about the shared
dataset or training loop needs to change.
"""

from typing import Any, Dict, Optional

from model.device import setup_backend

setup_backend()

import keras  # noqa: E402
import keras_hub  # noqa: E402
from keras import layers  # noqa: E402

from model.interfaces import BaseLM  # noqa: E402


@keras.saving.register_keras_serializable(package="model")
class SimpleGPTLM(BaseLM):
    """Standard GPT-2-style decoder-only Transformer.

    Forward pass:
        1. Token embedding + learned position embedding.
        2. `num_layers` `keras_hub.layers.TransformerDecoder` blocks
           (Pre-LN, causal self-attention, GELU feedforward).
        3. Final `LayerNormalization`.
        4. Take the last context position only.
        5. Plain `Dense(vocabulary_size)` readout -> logits.

    Args:
        vocabulary_size: Token vocabulary size (width of the logits).
        dim: Model/embedding dimension.
        window_size: Context length.
        num_layers: Number of `TransformerDecoder` blocks.
        num_heads: Attention heads per block.
        mlp_dim: Feedforward network hidden size (`intermediate_dim`).
        dropout: Dropout used inside attention and the feedforward network.
    """

    def __init__(
        self,
        vocabulary_size: int = 10000,
        dim: int = 128,
        window_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if vocabulary_size <= 0:
            raise ValueError(f"`vocabulary_size` must be positive, got {vocabulary_size}")
        if dim <= 0:
            raise ValueError(f"`dim` must be positive, got {dim}")
        if window_size <= 0:
            raise ValueError(f"`window_size` must be positive, got {window_size}")
        if num_layers <= 0:
            raise ValueError(f"`num_layers` must be positive, got {num_layers}")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"`dim` ({dim}) must be divisible by `num_heads` ({num_heads})")
        if mlp_dim <= 0:
            raise ValueError(f"`mlp_dim` must be positive, got {mlp_dim}")

        self._vocabulary_size = int(vocabulary_size)
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.mlp_dim = int(mlp_dim)
        self.dropout = float(dropout)

        self.token_embedding = layers.Embedding(self._vocabulary_size, self.dim, name="token_emb")
        self.position_embedding = keras_hub.layers.PositionEmbedding(
            sequence_length=self.window_size, name="pos_emb"
        )
        self.decoder_blocks = [
            keras_hub.layers.TransformerDecoder(
                intermediate_dim=self.mlp_dim,
                num_heads=self.num_heads,
                activation="gelu",
                dropout=self.dropout,
                normalize_first=True,
                name=f"decoder_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.final_norm = layers.LayerNormalization(name="final_norm")
        self.readout = layers.Dense(self._vocabulary_size, name="readout")

        self.build()

    # -- BaseLM contract ------------------------------------------------------

    @property
    def context_length(self) -> int:
        return self.window_size

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    # -- weights --------------------------------------------------------------

    def build(self, input_shape: Optional[Any] = None) -> None:
        if self.built:
            return
        self.token_embedding.build((None, self.window_size))
        self.position_embedding.build((None, self.window_size, self.dim))
        for block in self.decoder_blocks:
            block.build((None, self.window_size, self.dim))
        self.final_norm.build((None, self.window_size, self.dim))
        self.readout.build((None, self.dim))
        self.built = True

    # -- forward --------------------------------------------------------------

    def call(self, inputs: Any, training: bool = False) -> Any:
        token_ids = keras.ops.cast(inputs, "int32")

        tok = self.token_embedding(token_ids)
        pos = self.position_embedding(tok)
        x = tok + pos

        for block in self.decoder_blocks:
            x = block(x, use_causal_mask=True, training=training)

        x = self.final_norm(x, training=training)
        last = x[:, -1, :]
        return self.readout(last, training=training)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "vocabulary_size": self._vocabulary_size,
            "dim": self.dim,
            "window_size": self.window_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
        }
        return {**super().get_config(), **config}
