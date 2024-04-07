import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import numpy as np
from typing import List

resnet_kernel_init = nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal')

# class ResidualBlock(nn.Module):
#     out_channels: int
#     subsample: bool = False

#     @nn.compact
#     def __call__(self, x, train=True):
#         y = nn.Conv(self.out_channels, kernel_size=(3, 3),
#                     strides=(1, 1),
#                     kernel_init=resnet_kernel_init,
#                     use_bias=False)(x)
#         y = nn.BatchNorm()(y, use_running_average=not train)
#         y = nn.relu(y)
#         y = nn.Conv(self.out_channels, kernel_size=(3, 3),
#                     kernel_init=resnet_kernel_init,
#                     use_bias=False)(y)
#         y = nn.BatchNorm()(y, use_running_average=not train)

#         if self.subsample:
#             x = nn.Conv(self.out_channels, kernel_size=(1, 1), strides=(2, 2), kernel_init=resnet_kernel_init)(x)

#         x_out = nn.relu(y + x)
#         return x_out

# class ResidualBlock(nn.Module):
#     out_channels: int

#     @nn.compact
#     def __call__(self, x):
#         residual = x
#         x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
#         x = nn.BatchNorm()(x)
#         x = nn.relu(x)
#         x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
#         x = nn.BatchNorm()(x)
#         x = x + residual  # Residual connection
#         x = nn.relu(x)
#         return x

class ResidualBlock(nn.Module):
    out_channels: int
    stride: int
    subsample: bool

    @nn.compact
    def __call__(self, x, train=True):
        y = nn.Conv(features=self.out_channels, kernel_size=3, kernel_init=resnet_kernel_init, strides=self.stride, padding=1, use_bias=False)(x)
        y = nn.BatchNorm(dtype=jnp.float32, use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(features=self.out_channels, kernel_size=3, kernel_init=resnet_kernel_init, strides=1, padding=1, use_bias=False)(y)
        y = nn.BatchNorm(dtype=jnp.float32, use_running_average=not train)(y)
        if self.subsample:
            i = nn.Conv(features=self.out_channels, kernel_size=1, kernel_init=resnet_kernel_init, strides=self.stride, use_bias=False)(x)
            i = nn.BatchNorm(dtype=jnp.float32, use_running_average=not train)(i)
        else:
            i = x
        y += i
        y = nn.relu(y)
        return y

class ResidualLayer(nn.Module):
    out_channels: int
    stride: int
    num_blocks: int

    res_blocks: List[ResidualBlock]

    def setup(self):
        self.res_blocks.append(ResidualBlock(out_channels=self.out_channels, stride=self.stride, downsample=True))
        for _ in range(self.num_blocks):
            self.res_blocks.append(ResidualBlock(sout_channels=self.out_channels, stride=1, downsample=False))

    @nn.compact
    def __call__(self, x):
        y = x
        for res_block in self.res_blocks:
            y = res_block(y)
        return y
