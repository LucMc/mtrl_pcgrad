import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from mtrl.config.nn import MOOREConfig

from .base import MLP


class OrthogonalLayer1D(nn.Module):
    num_experts: int

    @nn.compact
    def __call__(
        self, x: Float[Array, "batch_size num_experts dim"]
    ) -> Float[Array, "batch_size num_experts dim"]:
        chex.assert_rank(x, 3)

        basis = jnp.expand_dims(
            x[:, 0, :] / jnp.linalg.norm(x[:, 0, :], axis=1, keepdims=True), axis=1
        )

        for i in range(1, self.num_experts):
            v = jnp.expand_dims(x[:, i, :], axis=1)  # (batch_size, 1, dim)
            w = v - ((v @ basis.transpose(0, 2, 1)) @ basis)
            wnorm = w / jnp.linalg.norm(w, axis=2, keepdims=True)
            basis = jnp.concatenate((basis, wnorm), axis=1)

        chex.assert_equal_shape((x, basis))

        return basis


class MOORENetwork(nn.Module):
    config: MOOREConfig

    head_dim: int  # o, 1 for Q networks and 2 * action_dim for policy networks
    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        batch_dim = x.shape[0]
        assert self.config.num_tasks is not None, "Number of tasks must be provided."
        task_idx = x[..., -self.config.num_tasks :]
        x = x[..., : -self.config.num_tasks]

        # Task ID embedding
        task_embedding = nn.Dense(
            self.config.num_experts,
            use_bias=self.config.use_bias,
            kernel_init=self.config.kernel_init(),
            bias_init=self.config.bias_init(),
        )(task_idx)

        experts_out = nn.vmap(
            MLP,
            variable_axes={"params": 0, "intermediates": 1},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,  # pyright: ignore [reportArgumentType]
            out_axes=-2,
            axis_size=self.config.num_experts,
        )(
            self.config.width,
            self.config.depth,
            self.config.width,
            self.config.activation,
            self.config.kernel_init(),
            self.config.bias_init(),
            self.config.use_bias,
        )(x)
        experts_out = OrthogonalLayer1D(self.config.num_experts)(experts_out)
        features_out = jnp.einsum("bnk,bn->bk", experts_out, task_embedding)
        features_out = self.config.activation(features_out)
        self.sow("intermediates", "torso_output", features_out)

        # MH
        x = nn.vmap(
            nn.Dense,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,  # pyright: ignore [reportArgumentType]
            out_axes=1,
            axis_size=self.config.num_tasks,
        )(
            self.head_dim,
            kernel_init=self.head_kernel_init,
            bias_init=self.head_bias_init,
            use_bias=self.config.use_bias,
        )(x)

        task_indices = task_idx.argmax(axis=-1)
        x = x[jnp.arange(batch_dim), task_indices]

        return x
