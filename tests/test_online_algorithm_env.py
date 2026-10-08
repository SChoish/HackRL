from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from hackrl.online_algorithm_env import (
    DUAL,
    LEO,
    PQN,
    build_online_transitions,
    clone_training_branch,
    environment_adapter,
    frozen_evaluation_binding,
    OnlineValueNetworks,
)


def _inputs(value):
    return (
        jnp.full((2, 3, 3, 1), value, dtype=jnp.float32),
        jnp.full((2, 4), value, dtype=jnp.float32),
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )


def test_environment_adapters_use_canonical_action_and_goal_counts():
    tick = environment_adapter("tick-claim")
    pack = environment_adapter("PACK_RESTORE")
    assert tick.name == "TICK-CLAIM"
    assert pack.name == "PACK-RESTORE"
    assert tick.num_actions > 1 and pack.num_actions > 1
    assert tick.num_goals == pack.num_goals == 12
    with pytest.raises(ValueError):
        environment_adapter("mine")


def test_transition_translation_keeps_commanded_and_all_goal_termination_separate():
    adapter = environment_adapter("tick")
    event = SimpleNamespace(
        valid_transition=jnp.asarray([True, False]),
        reward=jnp.asarray([1.0, 0.0]),
        goal_done=jnp.asarray([True, False]),
        world_done=jnp.asarray([False, True]),
        terminal_goals=jnp.asarray([[True, False], [False, False]]),
    )
    pair = build_online_transitions(
        adapter,
        _inputs(0.0),
        jnp.asarray([1, 2]),
        event,
        _inputs(1.0),
    )
    np.testing.assert_array_equal(pair.pqn.done, [True, True])
    np.testing.assert_array_equal(pair.pqn.valid, [True, False])
    np.testing.assert_array_equal(pair.leo.world_done, [False, True])
    np.testing.assert_array_equal(pair.leo.terminal_goals, event.terminal_goals)
    np.testing.assert_allclose(pair.pqn.next_map_channels, 1.0)


def test_clone_materializes_identical_independent_array_leaves():
    original = {
        "rng": jnp.asarray([1, 2], dtype=jnp.uint32),
        "replay": jnp.arange(4, dtype=jnp.float32),
    }
    cloned = clone_training_branch(original)
    np.testing.assert_array_equal(cloned["rng"], original["rng"])
    np.testing.assert_array_equal(cloned["replay"], original["replay"])
    changed = {**cloned, "replay": cloned["replay"].at[0].set(99.0)}
    assert float(original["replay"][0]) == 0.0
    assert float(changed["replay"][0]) == 99.0


def test_method_names_keep_dual_q_combination_distinct_from_bc():
    assert (PQN, LEO, DUAL) == ("GC-PQN", "LEO", "Dual LEO(PQN)")
    assert "BC" not in DUAL


def test_q_evaluation_rejects_accidental_softmax_sampling():
    state = {"placeholder": jnp.asarray(0)}
    network, returned = frozen_evaluation_binding(
        method=PQN,
        networks=OnlineValueNetworks(pqn=object()),
        state=state,
        stochastic=False,
    )
    assert network.method == PQN
    assert returned is state
    with pytest.raises(ValueError, match="epsilon-behavior"):
        frozen_evaluation_binding(
            method=PQN,
            networks=OnlineValueNetworks(pqn=object()),
            state=state,
            stochastic=True,
        )
