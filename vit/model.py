import jax
from jax import (
    Array,
    numpy as jnp,
)
import keras as nn

from .ops import (
    patchify, 
    PositionalEmbedding, 
    TransformerBlock
)

class VIT(nn.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.patchify = lambda x: patchify(x, config.patch_size)
        self.P = config.patch_size
        self.N = config.H*config.W//config.patch_size

        self.pos_embed = PositionalEmbedding(
            maxlen=config.maxlen,
            dim=config.d_model
        ).sinusoidal_embeddings() # (1, 1+N, d_model)

        self.proj_flattened_patches = nn.layers.Dense(
            config.d_model
        )
        self.class_emb = self.add_weight(shape=(1, config.d_model))

        self.encoder_layers = [
            TransformerBlock(causal=False, config=config)
            for _ in range(config.num_layers)
        ]
        self.norm = nn.layers.LayerNormalization(epsilon=1e-5)
        self.mlp_head = nn.layers.Dense(config.num_classes)

    def call(self, x:Array, training:bool=True): # (B, H, W, C)
        x = self.patchify(x) # (B, N, (P**2)*C)
        x = self.proj_flattened_patches(x) # (B, N, d_model)
        x = jnp.concatenate(
            (
                jnp.broadcast_to(self.class_emb.value, (x.shape[0], 1, x.shape[-1])),
                x
            ),
            axis=1
        ) # (B, 1+N, d_model)
        x += self.pos_embed # (B, 1+N, d_model)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training) # (B, 1+N, d_model)

        class_rep = x[:, 0, :] # (B, d_model)
        x = self.norm(class_rep) # (B, d_model)
        x = self.mlp_head(x) # (B, num_classes)
        return x # logits
    
