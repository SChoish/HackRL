import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreSplit,
    PackRestoreStart,
    PackRestoreVariant,
    make_pack_restore_state,
    pack_restore_step,
)
from hackrl.pack_restore_gc import (
    PackRestoreGCConfig,
    config_from_pack_restore_gc_payload,
    initialize_pack_restore_gc,
    load_pack_restore_gc_checkpoint,
    make_pack_restore_gc_update,
    pack_restore_defect_keys,
    pack_restore_gc_config_payload,
    save_pack_restore_gc_checkpoint,
    step_pack_restore_gc_workers,
)


def _ready(carried=2, remaining=2):
    state = make_pack_restore_state(
        0,
        int(PackRestorePhase.EMPTY),
        split=PackRestoreSplit.VALIDATION,
        start=PackRestoreStart.PATH_CHECK,
        source_growth_period=8,
    )
    state = pack_restore_step(
        state, int(PackRestoreAction.RIGHT), PackRestoreVariant.FIXED
    )
    return state.replace(
        carried_grain=jnp.asarray(carried, dtype=jnp.int32),
        source_grain=jnp.asarray(remaining, dtype=jnp.int32),
        source_age=jnp.asarray(8, dtype=jnp.int32),
    )


def _runner_at_trigger(config):
    _, runner = initialize_pack_restore_gc(config)
    state = _ready()
    batch = jax.tree.map(lambda value: jnp.stack((value, value)), state)
    return runner.replace(
        env_state=batch,
        command_active=jnp.ones((2,), dtype=jnp.bool_),
        env_steps=jnp.asarray(128, dtype=jnp.int32),
    )


def _config(probability):
    return PackRestoreGCConfig(
        variant=PackRestoreVariant.INVENTORY_OVERWRITE.value,
        inventory_overwrite_probability=probability,
        seed=17,
        num_envs=2,
        num_steps=2,
        minibatch_size=4,
        num_updates=1,
        policy_hidden_size=32,
        teacher_hidden_size=32,
        goal_mode="deliver_3",
    )


def test_stochastic_worker_step_matches_fixed_and_harmful_endpoints():
    actions = jnp.full((2,), int(PackRestoreAction.DO), dtype=jnp.int32)
    fixed_config = _config(0.0)
    harmful_config = _config(1.0)
    fixed_runner = _runner_at_trigger(fixed_config)
    harmful_runner = _runner_at_trigger(harmful_config)

    fixed_after, fixed_event = step_pack_restore_gc_workers(
        fixed_runner, actions, fixed_config
    )
    harmful_after, harmful_event = step_pack_restore_gc_workers(
        harmful_runner, actions, harmful_config
    )
    assert np.all(np.asarray(fixed_event.overwrite_trigger))
    assert not np.any(np.asarray(fixed_event.overwrite_fired))
    assert not np.any(np.asarray(fixed_event.physical_loss))
    assert np.all(np.asarray(harmful_event.overwrite_trigger))
    assert np.all(np.asarray(harmful_event.overwrite_fired))
    assert np.all(np.asarray(harmful_event.lost_grain) == 2)
    assert np.all(np.asarray(fixed_after.env_state.carried_grain) == 3)
    assert np.all(np.asarray(harmful_after.env_state.carried_grain) == 1)


def test_defect_keys_are_stable_and_slot_indexed():
    indices = jnp.asarray([0, 1, 512, 513], dtype=jnp.int32)
    first = np.asarray(pack_restore_defect_keys(30, indices))
    second = np.asarray(pack_restore_defect_keys(30, indices))
    other_seed = np.asarray(pack_restore_defect_keys(31, indices))
    assert np.array_equal(first, second)
    assert len({row.tobytes() for row in first}) == len(indices)
    assert not np.array_equal(first, other_seed)


def test_probability_config_validates_and_old_payload_defaults_to_p1():
    with pytest.raises(ValueError):
        replace(_config(0.05), inventory_overwrite_probability=-0.01).validate()
    with pytest.raises(ValueError):
        replace(_config(0.05), inventory_overwrite_probability=1.01).validate()

    payload = pack_restore_gc_config_payload(_config(1.0))
    payload.pop("inventory_overwrite_probability")
    restored = config_from_pack_restore_gc_payload(payload)
    assert restored.inventory_overwrite_probability == 1.0


def test_stochastic_update_and_checkpoint_round_trip(tmp_path):
    config = replace(
        _config(0.05),
        num_steps=4,
        minibatch_size=8,
        goal_mode="deliver_3",
    )
    network, runner = initialize_pack_restore_gc(config)
    updated, metrics = jax.jit(make_pack_restore_gc_update(network, config))(
        runner
    )
    assert int(updated.global_update) == 1
    assert int(metrics["transitions"]) == 8
    assert "overwrite_trigger_events" in metrics
    assert "overwrite_fire_events" in metrics
    destination = tmp_path / "update_1"
    save_pack_restore_gc_checkpoint(destination, updated, config)
    payload = json.loads(
        (destination / "config.json").read_text(encoding="utf-8")
    )
    assert payload["inventory_overwrite_probability"] == 0.05
    _, template = initialize_pack_restore_gc(config)
    restored = load_pack_restore_gc_checkpoint(destination, template, config)
    assert int(restored.global_update) == 1
