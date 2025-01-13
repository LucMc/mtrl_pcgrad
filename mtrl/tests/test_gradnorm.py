from doctest import debug
from typing import final
from numpy.random import permutation
import time
import pytest
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import random
import numpy as np
from itertools import permutations
from functools import partial

from mtrl.envs.base import EnvConfig
from mtrl.rl.algorithms.mtsac import MTSACConfig, MTSAC
from mtrl.rl.algorithms.gradnorm import GradNormConfig, GradNorm, GradNormWeights, extract_task_weights

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig

from mtrl.config.nn import MultiHeadConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig
from mtrl.envs import MetaworldConfig
from mtrl.experiment import Experiment

from pprint import pprint
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict
import matplotlib.pyplot as plt
import matplotlib

from mtrl.rl.algorithms import (
    Algorithm,
    OffPolicyAlgorithm,
    get_algorithm_for_config,
)

SEED = 0

def get_task_grad(task_idx: int) -> Array:
    task_mask = task_ids[:, task_idx] == 1
    
    def task_loss(params: FrozenDict) -> Float[Array, ""]:
        q_pred = critic.apply_fn(params, data.observations, data.actions)
        loss = 0.5 * ((q_pred - next_q_value) ** 2 * task_mask[:, None]).mean()
        return loss
        
    grad = jax.grad(task_loss)(critic.params)
    flat_grad, _ = jax.flatten_util.ravel_pytree(grad)
    return flat_grad


class TestGradNorm:
    
    def get_alg_cls(self, asymmetry: float, gn_only=False):
        SEED = 0

        # GRADNORM CLS
        algorithm_conf = GradNormConfig(
            asymmetry=asymmetry, # Change for testing
            num_tasks=10,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=MultiHeadConfig(
                    num_tasks=10, optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=MultiHeadConfig(
                    num_tasks=10,
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                )
            ),
            num_critics=2,
            use_task_weights=True, # Turn on when I've updated the task weights update blank pass thru
            gn_optimizer_config=OptimizerConfig(lr=3e-4) # Just for testing (normally 3e-4)
            
        )
        env = MetaworldConfig(
            env_id="MT10",
            terminate_on_success=False,
        ) 
        env.spawn(seed=SEED)

        gradnorm_cls = get_algorithm_for_config(algorithm_conf)
        gradnorm_cls = gradnorm_cls.initialize(algorithm_conf, env, seed=SEED)
        if gn_only: return None, gradnorm_cls

        # MTSAC CLS
        algorithm_conf = MTSACConfig(
            num_tasks=10,
            gamma=0.99,
            actor_config=ContinuousActionPolicyConfig(
                network_config=MultiHeadConfig(
                    num_tasks=10, optimizer=OptimizerConfig(max_grad_norm=1.0)
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=MultiHeadConfig(
                    num_tasks=10,
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                )
            ),
            num_critics=2,
            use_task_weights=False,
        )
        env.spawn(seed=SEED)
        mtsac_cls = get_algorithm_for_config(algorithm_conf)
        mtsac_cls = mtsac_cls.initialize(algorithm_conf, env, seed=SEED)

        return mtsac_cls, gradnorm_cls
    
    def plot_task_histogram(self, task_ids):
        matplotlib.use('QtAgg')
        plt.hist(jnp.argmax(task_ids, axis=1))
        plt.show()

    def gen_experience(self, alg_cls, config):
        # spawn off-policy replay buffer
        env_config = MetaworldConfig(
                env_id="MT10",
                terminate_on_success=False
                )
        replay_buffer = alg_cls.spawn_replay_buffer(env_config, config, SEED)

        # Generate some random experience
        env = env_config.spawn(seed=SEED)
        obs, _ = env.reset(seed=SEED)

        episodic_rew = []
        episodic_len = []
        episodes_ended = 0
        
        # Envs have autoresetting
        for ts in range(20_000):
            actions = env.action_space.sample()
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            replay_buffer.add(obs, next_obs, actions, rewards, terminations)

            has_autoreset = np.logical_or(terminations, truncations)
            for i, env_ended in enumerate(has_autoreset):
                if env_ended:
                    episodic_rew.append(int(infos["episode"]["r"][i]))
                    episodic_len.append(int(infos["episode"]["l"][i]))
                    episodes_ended += 1

            obs = next_obs
        return replay_buffer

    # def test_critic_uses_weights():
    #     data = replay_buffer.sample(config.batch_size)
    #     gradnorm_cls, logs, original_losses = gradnorm_cls.update(data, original_losses)
    '''
    1. Test when alpha=0 it is the same loss as MTSAC
    2. Test the grads of a task doing worse are emphasised
    3. Test the grads of a task doing better are lower
    4. Anything else...
    '''

    def overfit_alg(self, asymmetry, config, debug=True, epochs=30, gn_only=False):
        mtsac_cls, gradnorm_cls = self.get_alg_cls(asymmetry=asymmetry, gn_only=gn_only)
        # Just overfit MTSAC/GradNorm on some experience to see loss decrease

        replay_buffer = self.gen_experience(gradnorm_cls, config)
        original_losses = jnp.full((gradnorm_cls.num_tasks,), jnp.nan)
        task_layer = GradNormWeights(gradnorm_cls.num_tasks)
        # task_layer = nn.Dense(features=gradnorm_cls.num_tasks, use_bias=False)
        task_weights = task_layer.init(gradnorm_cls.key, jnp.ones(gradnorm_cls.num_tasks))

        # train GradNormSAC on that experience
        # test_critic_uses_weights()
        gn_losses = []

        print("GradNorm overfitting to experience test")
        print("asymmetry:\n", gradnorm_cls.asymmetry)
        for epoch in range(epochs):
            data = replay_buffer.sample(config.batch_size)
            
            # uncomment for debugging
            gradnorm_cls.gn_state.params['params']['gn_weights'] = jnp.ones(gradnorm_cls.num_tasks)
            if debug:
                with jax.disable_jit(): gradnorm_cls, logs, original_losses = gradnorm_cls.update(data, original_losses)
            else:
                gradnorm_cls, logs, original_losses = gradnorm_cls.update(data, original_losses)

            gn_losses.append(logs["losses/qf_loss"])
            print(f"{epoch} GN epoch - qf loss: {logs["losses/qf_loss"]}")

        if gn_only: return None, (gn_losses, gradnorm_cls), replay_buffer

        # train MTMH on that experience
        mtsac_losses = []
        print("MTMHSAC")
        for epoch in range(epochs):
            data = replay_buffer.sample(config.batch_size)
            mtsac_cls, logs = mtsac_cls.update(data)
            print(f"{epoch} SAC epoch - qf loss: {logs["losses/qf_loss"]}")
            mtsac_losses.append(logs["losses/qf_loss"])

        return (mtsac_losses, mtsac_cls), (gn_losses, gradnorm_cls), replay_buffer


    def test_vmap_task_grads(self):

        config = OffPolicyTrainingConfig(
                total_steps=int(2e7),
                buffer_size=int(1e6),
                batch_size=1280,
                warmstart_steps=1000
                )

        _, (gn_losses1, gn_cls), rb = self.overfit_alg(asymmetry=1,config=config, debug=True, epochs=5, gn_only=True)
        data = rb.sample(config.batch_size)
        critic = gn_cls.critic
        key, actor_loss_key, critic_loss_key = jax.random.split(gn_cls.key, 3)

        task_ids = jnp.array(data.observations[..., -gn_cls.num_tasks :])
        # self.plot_task_histogram(task_ids)
        task_weights = extract_task_weights(gn_cls.gn_state.params, task_ids) # Get normalised task weights
        alpha_val = 0.1 # Dummy as not important for test

        # Sample a'
        next_actions, next_action_log_probs = gn_cls.actor.apply_fn(
            gn_cls.actor.params, data.next_observations
        ).sample_and_log_prob(seed=critic_loss_key)

        # Compute target Q values
        q_values = gn_cls.critic.apply_fn(
            gn_cls.critic.target_params, data.next_observations, next_actions
        )
        min_qf_next_target = jnp.min(
            q_values, axis=0
        ) - alpha_val * next_action_log_probs.reshape(-1, 1)
        next_q_value = jax.lax.stop_gradient(
            data.rewards + (1 - data.dones) * gn_cls.gamma * min_qf_next_target
        )
        def critic_loss(
            params: FrozenDict,
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            # next_action_log_probs is (B,) shaped because of the sum(axis=1), while Q values are (B, 1)
            min_qf_next_target = jnp.min(
                q_values, axis=0
            ) - alpha_val * next_action_log_probs.reshape(-1, 1)
            next_q_value = jax.lax.stop_gradient(
                data.rewards + (1 - data.dones) * gn_cls.gamma * min_qf_next_target
            )

            q_pred = gn_cls.critic.apply_fn(params, data.observations, data.actions)
            if gn_cls.use_task_weights:
                assert task_weights is not None
                loss = (
                    0.5
                    * (jax.lax.stop_gradient(task_weights) * (q_pred - next_q_value) ** 2)
                    .mean(axis=1)
                    .sum()
                )
            else:
                loss = 0.5 * ((q_pred - next_q_value) ** 2).mean(axis=1).sum()
            return loss, q_pred.mean()

        def vmapped_task_losses(critic, data, task_weights, gn_cls, next_q_value):
            # Get grads and loss for each task
            def get_task_grad(task_idx: int, task_weights: Array) -> Array:
                task_mask = task_ids[:, task_idx] == 1
                
                def task_loss(params: FrozenDict) -> Float[Array, ""]:
                    q_pred = critic.apply_fn(params, data.observations, data.actions)
                    loss = (
                        0.5
                        * (task_weights * (q_pred - next_q_value) ** 2)
                        .mean(axis=1)
                        .sum()
                    )
                    return loss

                # Get gradients for each task using vmap
                loss, grad = jax.value_and_grad(task_loss)(critic.params)
                flat_grad, _ = jax.flatten_util.ravel_pytree(grad)
                return flat_grad, loss

            task_grads, task_losses = jax.vmap(get_task_grad, in_axes=(0,None))(jnp.arange(gn_cls.num_tasks).reshape(-1,1), task_weights)
            print("task_losses:\n", task_losses)
            return task_losses

        def looped_task_losses(critic, data, task_weights, gn_cls, next_q_value):
            task_losses = []
            task_grads = []

            for task_idx in range(gn_cls.num_tasks):
                # Create mask for current task
                task_mask = task_ids[:, task_idx] == 1
                
                def task_loss(params: FrozenDict) -> Float[Array, ""]:
                    q_pred = critic.apply_fn(params, data.observations, data.actions)
                    loss = (
                        0.5 
                        * (task_weights * (q_pred - next_q_value) ** 2)
                        .mean(axis=1)
                        .sum()
                    )
                    return loss

                # Get gradients and loss for current task
                loss, grad = jax.value_and_grad(task_loss)(critic.params)
                flat_grad, _ = jax.flatten_util.ravel_pytree(grad)
        
                task_grads.append(flat_grad)
                task_losses.append(loss)

            # Convert lists to JAX arrays
            task_grads = jnp.stack(task_grads)
            task_losses = jnp.array(task_losses)

            print("task_losses:\n", task_losses)
            return task_losses

        (critic_loss_value, qf_values), critic_grads = jax.value_and_grad(
            critic_loss, has_aux=True
        )(critic.params)

        vmapped_loss = vmapped_task_losses(critic, data, task_weights, gn_cls, next_q_value)
        looped_loss = looped_task_losses(critic, data, task_weights, gn_cls, next_q_value)
        assert jnp.all(vmapped_loss == looped_loss)
        assert vmapped_loss == critic_loss.mean()


    def test_overfit(self):
        config = OffPolicyTrainingConfig(
                total_steps=int(2e7),
                buffer_size=int(1e6),
                batch_size=1280,
                warmstart_steps=1000
                )
        pass
        # sac_losses1, gn_losses1 = self.overfit_alg(asymmetry=1, config=config, debug=True, epochs=5)
        # sac_losses2, gn_losses2 = self.overfit_alg(asymmetry=1, config=config, debug=False, epochs=5)
        #
        # # Check seeding works
        # assert sac_losses1 == sac_losses2
        # assert gn_losses1 == gn_losses2
