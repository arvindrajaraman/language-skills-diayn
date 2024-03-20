import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import numpy as np

resnet_kernel_init = nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal')

class ResidualBlock(nn.Module):
    out_channels: int
    subsample: bool = False

    @nn.compact
    def __call__(self, x, train=True):
        y = nn.Conv(self.out_channels, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(x)
        y = nn.BatchNorm()(y, use_running_average=not train)
        y = nn.relu(y)
        y = nn.Conv(self.out_channels, kernel_size=(3, 3),
                    kernel_init=resnet_kernel_init,
                    use_bias=False)(y)
        y = nn.BatchNorm()(y, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(self.out_channels, kernel_size=(1, 1), strides=(2, 2), kernel_init=resnet_kernel_init)(x)

        x_out = nn.relu(y + x)
        return x_out

class ResidualBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.BatchNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.BatchNorm()(x)
        x = x + residual  # Residual connection
        x = nn.relu(x)
        return x

class ResidualBlock(nn.Module):
    out_channels: int=
    stride: int

    @nn.compact
    def __call__(self, x, train=True):
        y = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=self.stride, padding="SAME")(x)
        y = nn.BatchNorm(dtype=jnp.float32, use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(features=self.out_channels, kernel_size=(3, 3), strides=1, padding="SAME")(y)
        y = nn.BatchNorm(dtype=jnp.float32, use_running_average=not train)(y)
        if self.in_channels != self.out_channels or self.stride != 1:
            x = nn.Conv(features=self.out_channels, kernel_size=(1, 1), strides=self.stride, padding="SAME")(x)
        y += x
        y = nn.relu(y)
        return y

class ResidualLayer(nn.Module):
    in_channels: int
    out_channels: int
    stride: int
    num_blocks: int

    @nn.compact
    def __call__(self, x):
        y = ResidualBlock(self.in_channels, self.out_channels, self.stride)(x)
        for _ in range(self.num_blocks):
            y = ResidualBlock(self.in_channels, self.out_channels, self.stride)()
        return y
