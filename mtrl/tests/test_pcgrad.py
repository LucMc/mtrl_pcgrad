import pytest
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import random
import numpy as np

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

def get_task_grad(task_idx: int) -> Array:
    task_mask = task_ids[:, task_idx] == 1
    
    def task_loss(params: FrozenDict) -> Float[Array, ""]:
        q_pred = critic.apply_fn(params, data.observations, data.actions)
        loss = 0.5 * ((q_pred - next_q_value) ** 2 * task_mask[:, None]).mean()
        return loss
        
    grad = jax.grad(task_loss)(critic.params)
    flat_grad, _ = jax.flatten_util.ravel_pytree(grad)
    return flat_grad


# # Get gradients for each task using vmap
# task_grads = jax.vmap(get_task_grad)(jnp.arange(num_tasks))

# PCGrad projection
def project_grad(grad_i: Array, grad_j: Array) -> Array:
    dot_product = jnp.sum(grad_i * grad_j)
    grad_conflicts = dot_product < 0 # for metrics
    return jnp.where(
        dot_product < 0,
        grad_i - (dot_product * grad_j) / (jnp.sum(grad_j * grad_j) + 1e-12),
        grad_i
    ), grad_conflicts

def pcgrad_loop(num_tasks, task_grads):
    final_grads = task_grads
    total_grad_conflicts = 0
    for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    f_grads, grad_conflicts = project_grad(final_grads[i], task_grads[j])
                    final_grads = final_grads.at[i].set(f_grads)
                    total_grad_conflicts += grad_conflicts

    return final_grads, total_grad_conflicts

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
        PCGrad_cls = PCGrad_cls.initialize(algorithm_conf, env, seed=SEED)

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


        return MTSAC_cls, PCGrad_cls

    def test_grad_surgery(self): # alg_cls
        '''Test the competing gradients are being normalisaed correctly'''

        seed = 0 # TODO get seed from cls somewhere
        random.seed(seed)
        np.random.seed(seed)

        # Test Case 1: Completely opposing gradients
        # Expected: g1 projected onto g2's normal plane should be [0.0, 0.0]
        g1 = jnp.array([1.0, 0.0]) # dp = -1 (oposing directions)
        g2 = jnp.array([-1.0, 0.0])
        
        new_g1, conflicts = project_grad(g1,g2)
        
        assert jnp.array_equal(new_g1, jnp.array([0., 0.]))
        assert conflicts == 1 # Number of projections, either 0 or 1 in this simplke case


        # Test Case 2: Partially opposing gradients
        # Expected: g1 projected should preserve the non-conflicting component
        g1 = jnp.array([1.0, 1.0]) # dp = -1 + 1 = 0, thus orthogonal
        g2 = jnp.array([-1.0, 1.0])
        new_g1, conflicts = project_grad(g1,g2)
        
        assert jnp.array_equal(new_g1, jnp.array([1., 1.]))
        assert conflicts == 0


        # Test Case 3: Orthogonal gradients
        # Expected: g1 should remain unchanged as there's no conflict
        g1 = jnp.array([1.0, 0.0])
        g2 = jnp.array([0.0, 1.0])

        new_g1, conflicts = project_grad(g1,g2)
        assert jnp.array_equal(new_g1, jnp.array([1., 0.]))
        assert conflicts == 0

        # Test Case 4: Whole loop
        # Expected: (1) t1 conflicts t2 (2) t2 conflicts t1
        g  = jnp.array([
            [1.0, 0.0],   # Task 1
            [-1.0, 0.0],  # Task 2
            [0.0, 1.0]    # Task 3
        ])
        exp_g  = jnp.array([
            [0.0, 0.0],   # Task 1
            [0.0, 0.0],  # Task 2
            [0.0, 1.0]    # Task 3
        ])

        grads, conflicts = pcgrad_loop(3, g)
        assert conflicts == 2
        assert jnp.array_equal(grads, exp_g)



        # What is the gradient with normal SAC vs gradient with PCGrad?

        # data = replaybuffersamples
        # mtsac_updated_critic = mtsac.update_inner()
        # pcgrad_updated_critic = pcgrad.update_inner()

    def test_grad_diff(self):
        '''Test the gradients are different to regular SAC grads'''
        pass
