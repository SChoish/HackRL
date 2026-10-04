import jax
import jax.numpy as jnp
import numpy as np

from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreStart,
    PackRestoreVariant,
    make_pack_restore_state,
    pack_restore_step,
)
from hackrl.pack_restore_gc import (
    PackRestoreGCConfig,
    command_deliver_3,
    initialize_pack_restore_gc,
    load_pack_restore_gc_checkpoint,
    make_pack_restore_gc_update,
    save_pack_restore_gc_checkpoint,
    step_pack_restore_gc_workers,
)


def _ready_state():
    state = make_pack_restore_state(
        0, int(PackRestorePhase.LOADED), start=PackRestoreStart.PATH_CHECK
    )
    for action in (PackRestoreAction.MAKE_RECORD, PackRestoreAction.PACK_STORAGE):
        state = pack_restore_step(state, int(action), PackRestoreVariant.FIXED)
    return state


def test_rebuild_violation_is_recorded_only_for_the_mutant_kernel():
    ready = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), _ready_state())
    _, runner = initialize_pack_restore_gc(
        PackRestoreGCConfig(
            num_envs=2,
            num_steps=4,
            minibatch_size=8,
            num_updates=1,
            policy_hidden_size=32,
            goal_mode="deliver_3",
        )
    )
    runner = runner.replace(
        env_state=ready,
        command_active=jnp.ones((2,), dtype=jnp.bool_),
    )
    actions = jnp.full((2,), int(PackRestoreAction.REBUILD_EMPTY), dtype=jnp.int32)
    _, mutant = step_pack_restore_gc_workers(
        runner,
        actions,
        PackRestoreGCConfig(variant="mutant", num_envs=2, num_steps=4, minibatch_size=8),
    )
    _, fixed = step_pack_restore_gc_workers(
        runner,
        actions,
        PackRestoreGCConfig(variant="fixed", num_envs=2, num_steps=4, minibatch_size=8),
    )
    assert np.asarray(mutant.violation).tolist() == [True, True]
    assert np.asarray(fixed.violation).tolist() == [False, False]
    assert int(np.asarray(mutant.violation_grain_delivered).sum()) == 0


def test_update_and_checkpoint_round_trip(tmp_path):
    config = PackRestoreGCConfig(
        num_envs=2,
        num_steps=4,
        minibatch_size=8,
        num_updates=1,
        policy_hidden_size=32,
        goal_mode="workshop12",
        seed=0,
    )
    network, runner = initialize_pack_restore_gc(config)
    before = np.asarray(jax.device_get(jax.tree.leaves(runner.train_state.params)[0]))
    switched = command_deliver_3(runner)
    after = np.asarray(jax.device_get(jax.tree.leaves(switched.train_state.params)[0]))
    assert np.array_equal(before, after)
    updated, metrics = jax.jit(make_pack_restore_gc_update(network, config))(runner)
    assert int(updated.global_update) == 1
    assert int(metrics["valid_transitions"]) == 8
    destination = tmp_path / "update_1"
    save_pack_restore_gc_checkpoint(destination, updated, config)
    _, template = initialize_pack_restore_gc(config)
    restored = load_pack_restore_gc_checkpoint(destination, template, config)
    assert int(restored.global_update) == 1
    left = np.asarray(jax.device_get(jax.tree.leaves(updated.train_state.params)[0]))
    right = np.asarray(jax.device_get(jax.tree.leaves(restored.train_state.params)[0]))
    assert np.array_equal(left, right)
