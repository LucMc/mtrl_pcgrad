import pytest
import jax.numpy as jnp

from mtrl.envs.base import EnvConfig
from mtrl.rl.algorithms.mtsac import MTSACConfig, MTSAC
from mtrl.rl.algorithms.pcgrad import PCGradConfig, PCGrad

from mtrl.config.networks import ContinuousActionPolicyConfig, QValueFunctionConfig

from mtrl.config.nn import MultiHeadConfig
from mtrl.config.optim import OptimizerConfig
from mtrl.config.rl import OffPolicyTrainingConfig
from mtrl.envs import MetaworldConfig
from mtrl.experiment import Experiment

from mtrl.rl.algorithms import (
    Algorithm,
    OffPolicyAlgorithm,
    get_algorithm_for_config,
)


class TestPCGrad:
    @pytest.fixture
    def alg_cls(self):
        SEED = 0

        algorithm_conf = PCGradConfig(
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

        PCGrad_cls = get_algorithm_for_config(algorithm_conf)
        PCGrad_cls = algorithm_cls.initialize(algorithm_conf, env, seed=SEED)

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
        MTSAC_cls = algorithm_cls.initialize(algorithm_conf, env, seed=SEED)


        return MTSAC_cls, PCGrad_cls

    def test_grad_surgery(self, MTSAC_cls, PCGrad_cls):
        '''Test the competing gradients are being normalisaed correctly'''

        seed = 0 # TODO get seed from cls somewhere
        random.seed(seed)
        np.random.seed(seed)

        # Test Case 1: Completely opposing gradients
        g1 = jnp.array([1.0, 0.0])
        g2 = jnp.array([-1.0, 0.0])
        
        # Expected: g1 projected onto g2's normal plane should be [0.0, 0.0]

        # Test Case 2: Partially opposing gradients
        g1 = jnp.array([1.0, 1.0])
        g2 = jnp.array([-1.0, 1.0])
        # Expected: g1 projected should preserve the non-conflicting component

        # Test Case 3: Orthogonal gradients
        g1 = jnp.array([1.0, 0.0])
        g2 = jnp.array([0.0, 1.0])
        # Expected: g1 should remain unchanged as there's no conflict

        # Test Case 4: Multiple gradients
        grads = jnp.array([
            [1.0, 0.0],   # Task 1
            [-1.0, 0.0],  # Task 2
            [0.0, 1.0]    # Task 3
        ])

        # What is the gradient with normal SAC vs gradient with PCGrad?
        breakpoint()
        data = replaybuffersamples
        mtsac_updated_critic = mtsac.update_inner()
        pcgrad_updated_critic = pcgrad.update_inner()

        pass

    def test_grad_diff(self):
        '''Test the gradients are different to regular SAC grads'''
        pass
