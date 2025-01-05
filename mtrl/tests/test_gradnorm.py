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
from mtrl.rl.algorithms.gradnorm import GradNormConfig, GradNorm

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig

from mtrl.config.nn import MultiHeadConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig
from mtrl.envs import MetaworldConfig
from mtrl.experiment import Experiment

from pprint import pprint

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
    @pytest.fixture
    def alg_cls(self):
        SEED = 0

        # GRADNORM CLS
        algorithm_conf = GradNormConfig(
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
            use_task_weights=True,
        )
        env = MetaworldConfig(
            env_id="MT10",
            terminate_on_success=False,
        ) 
        env.spawn(seed=SEED)

        gradnorm_cls = get_algorithm_for_config(algorithm_conf)
        gradnorm_cls = gradnorm_cls.initialize(algorithm_conf, env, seed=SEED)

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
            use_task_weights=True,
        )
        env.spawn(seed=SEED)
        MTSAC_cls = get_algorithm_for_config(algorithm_conf)
        MTSAC_cls = MTSAC_cls.initialize(algorithm_conf, env, seed=SEED)

        return MTSAC_cls, gradnorm_cls

    '''
    1. Test when alpha=0 it is the same loss as MTSAC
    2. Test the grads of a task doing worse are emphasised
    3. Test the grads of a task doing better are lower
    4. Anything else...
    '''
    def test_gradnorm(self, alg_cls):
        MTSAC_cls, gradnorm_cls = alg_cls
        # Just overfit MTSAC/GradNorm on some experience to see loss decrease
        # spawn off-policy replay buffer
        config = OffPolicyTrainingConfig(
                total_steps=int(2e7),
                buffer_size=int(1e6),
                batch_size=1280,
                warmstart_steps=1000
                )

        env_config = MetaworldConfig(
                env_id="MT10",
                terminate_on_success=False
                )
        
        replay_buffer = gradnorm_cls.spawn_replay_buffer(env_config, config, SEED)

        # Generate some random experience
        env = env_config.spawn(seed=SEED)
        obs, _ = env.reset(seed=SEED)

        episodic_rew = []
        episodic_len = []
        episodes_ended = 0
        
        # Envs have autoresetting
        for ts in range(10_000):
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
            # print(f"Avg Reward: {rewards.mean()}")

        # print("episodic_rew:\n", *episodic_rew)
        # print("episodic_len:\n", *episodic_len)
        # print("episodes_ended:\n", episodes_ended)

        
        # train MTMH on that experience
        for epoch in range(30):
            data = replay_buffer.sample(config.batch_size)
            MTSAC_cls, logs = MTSAC_cls.update(data)
            print(f"{epoch} epoch - qf loss: {logs["losses/qf_loss"]}")


