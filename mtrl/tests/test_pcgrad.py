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
                    # print(i,j)
                    f_grads, grad_conflicts = project_grad(final_grads[i], task_grads[j])
                    # print(f_grads)
                    final_grads = final_grads.at[i].set(f_grads)
                    total_grad_conflicts += grad_conflicts
    
    print(final_grads)
    return final_grads, total_grad_conflicts/2

    # cs = []
    # for i in range(num_tasks):
    #     for j in range(num_tasks):
    #         if i!=j:
    #             cs.append((i,j))

def pcgrad_vmap_old(num_tasks, task_grads):
    final_grads = task_grads
    total_grad_conflicts = 0

    # project grad should take in a vector and output a vector
    def p_grads(i, final_grads, total_grad_conflicts): 
        for j in range(num_tasks):
            # Use a mask instead of if statement
            is_different = j != i # Is this nessessary since there shouldn't ever be a conflict with itself?
            f_grads, grad_conflicts = project_grad(final_grads[i], task_grads[j])
            
            # Only update when mask is True
            new_grads = jnp.where(is_different, f_grads, final_grads[i])
            new_conflicts = jnp.where(is_different, grad_conflicts, 0.0)

            # print("UPDATING", final_grads.at[i], "with",new_grads)
            final_grads = final_grads.at[i].set(new_grads)
            total_grad_conflicts += new_conflicts
            # print("UPDATED", final_grads.at[i])
            # print("new_grads", new_grads)

        return final_grads, total_grad_conflicts

    # with jax.disable_jit(): # I needs to be one at a time, but how?
    f_grads, g_conflicts = jax.vmap(p_grads, in_axes=(1,None,None))(jnp.arange(num_tasks).reshape(1,-1), final_grads, total_grad_conflicts)
    
    f_grads_exp = jnp.array([f_grads[0,0], # Use this to make work on tests
                            f_grads[1,1],
                            f_grads[2,2]])

    # print("f_grads old vmap:\n", f_grads)
    return f_grads_exp, sum(g_conflicts) 

def project_gradients(xy) -> jax.Array:
    x, y = xy
    dot = jnp.dot(x, y)
    grad_conflicts = dot < 0
    return jnp.where(grad_conflicts, x - (dot * y) / (jnp.sum(y**2) + 1e-8), x), grad_conflicts.sum()  # pyright: ignore[reportReturnType]

def pcgrad_vmap(num_tasks, task_grads):


    @partial(jax.vmap, in_axes=(0, 0, None), out_axes=0)
    def p_grads(
        task_gradient: Float[Array, " gradient_dim"],
        i: Float[Array, ""],
        all_grads: Float[Array, " num_tasks gradient_dim"],
    ) -> Float[Array, " gradient_dim"]:

        g_i_pc = task_gradient
        total = 0
        for j in range(all_grads.shape[0]):
           g_i_pc, confs = jax.lax.cond(i != j, (g_i_pc, all_grads[j]), project_gradients, g_i_pc, lambda x: (x,0))
           total += confs
        return g_i_pc, total

    # (num_tasks, gradient_dim)
    res, total = p_grads(task_grads, jnp.arange(num_tasks), task_grads)
    total = total.sum() / 2
    return res, total


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
        
        def test_loop(pcgrad_fn):
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

            grads, conflicts = pcgrad_fn(3, g)
            new_grads, new_conflicts = pcgrad_fn(3, grads)
            assert jnp.array_equal(new_grads, grads)
            assert new_conflicts == 0

            # print("conflicts:\n", conflicts)
            # print("exp_g:\n", exp_g)
            # print("grads:\n", grads)
            assert conflicts == 1
            assert jnp.array_equal(grads, exp_g)

            # Test Case 5: Partial conflicts in triangle
            # Expected: All tasks have some conflict but not complete cancellation
            g = jnp.array([
                [1.0, 1.0],    # Task 1 
                [-1.0, 1.0],   # Task 2 conflicts with 1
                [0.0, -1.0]    # Task 3 conflicts with 1
            ]) # Since each conf happens twice (T1/T2 and T2/T1) means 4 total

            exp_g = jnp.array([
                [1.0, 0.0],    
                [-1.0, 0.0],  
                [0.0, -0.0]  
            ])

            grads, conflicts = pcgrad_fn(3, g)
            assert conflicts == 2  # Three pairwise conflicts
            assert jnp.array_equal(grads, exp_g)
        
            new_grads, new_conflicts = pcgrad_fn(3, grads)
            # print("new_conflicts:\n", new_conflicts) # There is a grad conflict after??
            # assert jnp.array_equal(new_grads, grads)
            # assert new_conflicts == 0

        test_loop(pcgrad_loop)
        # test_loop(pcgrad_vmap_old)
        test_loop(pcgrad_vmap)

        # What is the gradient with normal SAC vs gradient with PCGrad?

        # data = replaybuffersamples
        # mtsac_updated_critic = mtsac.update_inner()
        # pcgrad_updated_critic = pcgrad.update_inner()

   ############################################################################## 
   ################################ METRICS ##################################### 
   ############################################################################## 

    def test_metrics(self):
        def testcases():

            testcases = []

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

            testcases.append((g, exp_g))

            # Test Case 5: Partial conflicts in triangle
            # Expected: All tasks have some conflict but not complete cancellation
            g = jnp.array([
                [1.0, 1.0],    # Task 1 
                [-1.0, 1.0],   # Task 2 conflicts with 1
                [0.0, -1.0]    # Task 3 conflicts with 1
            ]) # Since each conf happens twice (T1/T2 and T2/T1) means 4 total

            exp_g = jnp.array([
                [1.0, 0.0],    
                [-1.0, 0.0],  
                [0.0, -0.0]  
            ])
            testcases.append((g, exp_g))
            return testcases


        def metrics_loop(task_grads, final_grads, num_tasks):
            # Metrics
            avg_cos_sim = jnp.mean(
                    jnp.array([
                        jnp.sum(task_grads[i] * task_grads[j]) / (
                            jnp.linalg.norm(task_grads[i]) * jnp.linalg.norm(task_grads[j]) + 1e-12
                        )
                        for i in range(num_tasks)
                        for j in range(i + 1, num_tasks)
                    ]))

            new_cos_sim = jnp.mean(
                    jnp.array([
                        jnp.sum(final_grads[i] * final_grads[j]) / (
                            jnp.linalg.norm(final_grads[i]) * jnp.linalg.norm(final_grads[j]) + 1e-12
                        )
                        for i in range(num_tasks)
                        for j in range(i + 1, num_tasks)
                    ]))

            metrics = {
                # "metrics/pcgrad_n_grad_conflicts": total_grad_conflicts,
                "metrics/pcgrad_avg_critic_grad_magnitude": jnp.mean(jnp.linalg.norm(final_grads, axis=1)),
                "metrics/pcgrad_avg_critic_grad_magnitude_before_grad_surgery": jnp.mean(jnp.linalg.norm(task_grads, axis=1)),
                "metrics/pcgrad_avg_cosine_similarity":  avg_cos_sim,
            "metrics/pcgrad_avg_cosine_similarity_diff": avg_cos_sim - new_cos_sim
                
            }
            return metrics


        def metrics_vmap(task_grads, final_grads, num_taks):
            '''
            When calculating the cosine similarity between each task grad and final grad?
            Or cosine similarity between each task and all other tasks
             - Calculate cossine between one task and the next task
             - Caldulate vmap to get all the cos sims between all tasks
             - Use cond? Or tri or diag to basically just get the task A-> B since B->A is the same
            '''

            # cos_sim = jnp.array([ jnp.dot(task_grad, final_gradient) / 
            #                     (jnp.linalg.norm(task_gradient) * jnp.linalg.norm(task_grads[j]) + 1e-12)
            #
            #     ])

            avg_cos_sim = jnp.array([ # Removed the jnp.mean
                            jnp.sum(task_grads[i] * task_grads[j]) / (
                                    jnp.linalg.norm(task_grads[i]) * jnp.linalg.norm(task_grads[j]) + 1e-12
                                    )
                            for i in range(num_tasks)
                            for j in range(i + 1, num_tasks)
                            ])

            print("avg_cos_sim:\n", avg_cos_sim)

            def calc_cos_sim(task_gradient, compare_gradient, num_tasks):

                new_cos_sim = jnp.array([
                            jnp.sum(task_gradient * compare_gradient) / (
                                jnp.linalg.norm(task_gradient) * jnp.linalg.norm(compare_gradient) + 1e-12
                            )
                        ])

                avg_cos_sim = jnp.array([None])
                return avg_cos_sim, new_cos_sim

                # Loop through j's
                # jax.vmap(inner_loop)

            # Loop through i's
            with jax.disable_jit():
                inner_loop = jax.vmap(calc_cos_sim, in_axes=(0, 0, None))
                outer_loop = jax.vmap(inner_loop, in_axes=(0, 0, None))
                avg_cos_sim, new_cos_sim = outer_loop(task_grads, task_grads, num_taks) # Should be for every i,j
                avg_cos_sim, new_cos_sim = outer_loop(task_grads, final_grads, num_taks) # Should be for every i,j
                breakpoint()

            metrics = {
                # "metrics/pcgrad_n_grad_conflicts": total_grad_conflicts,
                "metrics/pcgrad_avg_critic_grad_magnitude": jnp.mean(jnp.linalg.norm(final_grads, axis=1)),
                "metrics/pcgrad_avg_critic_grad_magnitude_before_grad_surgery": jnp.mean(jnp.linalg.norm(task_grads, axis=1)),
                "metrics/pcgrad_avg_cosine_similarity":  avg_cos_sim,
            "metrics/pcgrad_avg_cosine_similarity_diff": avg_cos_sim - new_cos_sim
            }
            return metrics

        num_tasks = 3
        for g, exp_g in testcases():
            print("\nmetrics_loop:")
            print(metrics_loop(g, exp_g, num_tasks))

            print("\nmetrics_vmap:")
            print(metrics_vmap(g, exp_g, num_tasks))


    '''

    def p_grads(i): 
        def inner_loop(j, carry):
            final_grads, total_conflicts = carry
            # Use a mask instead of if statement
            is_different = j != i
            f_grads, grad_conflicts = project_grad(final_grads[i], task_grads[j])
            
            # Only update when mask is True
            new_grads = jnp.where(is_different, f_grads, final_grads[i])
            new_conflicts = jnp.where(is_different, grad_conflicts, 0.0)
            
            return (final_grads.at[i].set(new_grads), total_conflicts + new_conflicts)

    def p_grads(i): 
        def inner_loop(j, carry):
            final_grads, total_conflicts = carry
            # Use a mask instead of if statement
            is_different = j != i # Is this nessessary since there shouldn't ever be a conflict with itself?
            f_grads, grad_conflicts = project_grad(final_grads[i], task_grads[j])
            
            # Only update when mask is True
            new_grads = jnp.where(is_different, f_grads, final_grads[i])
            new_conflicts = jnp.where(is_different, grad_conflicts, 0.0)
            final_grads.at[i].set(new_grads)
            breakpoint()
            return (final_grads, total_conflicts + new_conflicts)

        # Initialize the carry value
        init_carry = (task_grads, 0.0)
        
        # Use scan instead of for loop
        final_grads, total_grad_conflicts = jax.lax.fori_loop(
            0, num_tasks, inner_loop, init_carry)
        breakpoint()
        
        return final_grads, total_grad_conflicts 

    f_grads, g_conflicts = jax.vmap(p_grads)(jnp.arange(num_tasks))
    
    breakpoint()
    print("f_grads:\n", f_grads)
    return f_grads, sum(g_conflicts) 

    '''
