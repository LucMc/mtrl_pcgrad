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
from mtrl.rl.algorithms.gradnorm import GradNormConfig, GradNorm, GradNormWeights

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig

from mtrl.config.nn import MultiHeadConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig
from mtrl.envs import MetaworldConfig
from mtrl.experiment import Experiment

from pprint import pprint
import flax.linen as nn

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
    
    def get_alg_cls(self, asymmetry: float):
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
        MTSAC_cls = get_algorithm_for_config(algorithm_conf)
        MTSAC_cls = MTSAC_cls.initialize(algorithm_conf, env, seed=SEED)

        return MTSAC_cls, gradnorm_cls
    

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
    def test_gradnorm(self):
        config = OffPolicyTrainingConfig(
                total_steps=int(2e7),
                buffer_size=int(1e6),
                batch_size=1280,
                warmstart_steps=1000
                )
        MTASC_cls, gradnorm_cls = self.get_alg_cls(asymmetry=0.1)
        # Just overfit MTSAC/GradNorm on some experience to see loss decrease

        replay_buffer = self.gen_experience(gradnorm_cls, config)
        original_losses = jnp.full((gradnorm_cls.num_tasks,), jnp.nan)
        task_layer = GradNormWeights(gradnorm_cls.num_tasks)
        # task_layer = nn.Dense(features=gradnorm_cls.num_tasks, use_bias=False)
        task_weights = task_layer.init(gradnorm_cls.key, jnp.ones(gradnorm_cls.num_tasks))

        # train GradNormSAC on that experience
        # test_critic_uses_weights()
        losses = []

        print("GradNorm overfitting to experience test")
        print("asymmetry:\n", gradnorm_cls.asymmetry)
        for epoch in range(30):
            data = replay_buffer.sample(config.batch_size)
            
            # uncomment for debugging
            with jax.disable_jit(): gradnorm_cls, logs, original_losses = gradnorm_cls.update(data, original_losses)
            # gradnorm_cls, logs, original_losses = gradnorm_cls.update(data, original_losses)
            losses.append(logs["losses/qf_loss"])
            print(f"{epoch} GN epoch - qf loss: {logs["losses/qf_loss"]}")

        # train MTMH on that experience
        # print("MTMHSAC")
        # for epoch in range(30):
        #     data = replay_buffer.sample(config.batch_size)
        #     MTSAC_cls, logs = MTSAC_cls.update(data)
        #     print(f"{epoch} SAC epoch - qf loss: {logs["losses/qf_loss"]}")


