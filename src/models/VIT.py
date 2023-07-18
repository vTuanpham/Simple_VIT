import time
from typing import Callable
import einops
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.linen.initializers import zeros, normal, uniform, lecun_normal


class InputEmbedding(nn.Module):
    latent_size: int
    patch_size: int
    n_channels: int
    batch_size: int

    class_token_init: Callable = lecun_normal()
    pos_embed_init: Callable = lecun_normal()

    @nn.compact
    def __call__(self, input_data):
        # Re-arrange image into patches.
        patches = einops.rearrange(
            input_data, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)
        linear_projection = nn.Dense(features=self.latent_size)(patches)
        class_token = self.param('class_token',
                                 self.class_token_init,
                                 (self.batch_size, 1, self.latent_size))
        b, n, _ = linear_projection.shape
        linear_projection = jnp.concatenate([class_token, linear_projection], axis=1)

        pos_embedding = self.param('pos_embedding',
                                 self.pos_embed_init,
                                 (self.batch_size, 1, self.latent_size))
        pos_embed = einops.repeat(pos_embedding, 'b 1 d -> b m d', m=n + 1)

        linear_projection += pos_embed

        return linear_projection


class EncoderBlock(nn.Module):
    latent_size: int
    num_heads: int
    dropout: float
    training: bool

    @nn.compact
    def __call__(self, embedded_patches):
        # First sublayer: Norm + Multi-Head Attention + residual connection.
        firstNorm_out = nn.LayerNorm(self.latent_size)(embedded_patches)
        # Multi-Head Attention layer
        attention_output = nn.MultiHeadDotProductAttention(self.num_heads,
                                                         dropout_rate=self.dropout,
                                                         deterministic=not self.training)(firstNorm_out, firstNorm_out)
        # First residual connection
        first_added_output = attention_output + embedded_patches

        # Second sublayer: Norm + enc_MLP (Feed forward)
        secondNorm_out = nn.LayerNorm(self.latent_size)(first_added_output)

        # MLP_head layer in the encoder. I use the same configuration as that
        # used in the original VitTransformer implementation. The ViT-Base
        # variant uses MLP_head size 3072, which is latent_size*4.
        ff_output = nn.Sequential([
            nn.Dense(self.latent_size * 4),
            nn.glu,
            nn.Dropout(self.dropout, deterministic=not self.training),
            nn.Dense(self.latent_size),
            nn.Dropout(self.dropout, deterministic=not self.training)
        ])(secondNorm_out)

        # Return the output of the second residual connection
        return ff_output + first_added_output


class VitTransformer(nn.Module):
    num_encoders: int
    latent_size: int
    patch_size: int
    n_channels: int
    batch_size: int
    num_heads: int
    dropout: float
    num_classes: int
    training: bool

    @nn.compact
    def __call__(self, input_img):
        # Apply input embedding (patchify + linear projection + position embeding)
        # to the input image passed to the model
        embed_input = InputEmbedding(self.latent_size,
                                    self.patch_size,
                                    self.n_channels,
                                    self.batch_size)(input_img)

        # Create a stack of encoder layers
        enc_output = nn.Sequential([EncoderBlock(self.latent_size,
                                                self.num_heads,
                                                self.dropout,
                                                self.training) for _ in range(self.num_encoders)])(embed_input)

        # Extract the output embedding information of the [class] token
        cls_token_embedding = enc_output[:, 0]

        # MLP_head at the classification stage has 'one hidden layer at pre-training time
        # and by a single linear layer at fine-tuning time'. For this implementation I will
        # use what was used for training, so I'll have a total of two layers, one hidden
        # layer and one output layer.
        output = nn.Sequential([
            nn.LayerNorm(self.latent_size),
            nn.Dense(self.latent_size),
            nn.Dense(self.num_classes)
        ])(cls_token_embedding)

        return output


if __name__ == "__main__":
    patch_size = 16  # Patch size (P) = 16
    latent_size = 768  # Latent vector (D). ViT-Base uses 768
    n_channels = 3  # Number of channels for input images
    num_heads = 12  # ViT-Base uses 12 heads
    num_encoders = 12  # ViT-Base uses 12 encoder layers
    dropout = 0.1  # Dropout = 0.1 is used with ViT-Base & ImageNet-21k
    num_classes = 10  # Number of classes in CIFAR10 dataset
    size = 224  # Size used for training = 224

    epochs = 30  # Number of epochs
    base_lr = 10e-3  # Base LR
    weight_decay = 0.03  # Weight decay for ViT-Base (on ImageNet-21k)
    batch_size = 4

    x_key, init_key = random.split(random.PRNGKey(42))
    test_input = random.uniform(x_key, (batch_size, n_channels, size, size))

    vision_transformer = VitTransformer(num_encoders,
                                        latent_size,
                                        patch_size,
                                        n_channels,
                                        batch_size,
                                        num_heads,
                                        dropout,
                                        num_classes,
                                        training=False)
    params = vision_transformer.init(init_key, test_input)
    output = vision_transformer.apply(params, test_input)

    print(output.shape)