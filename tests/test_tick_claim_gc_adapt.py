import jax

from hackrl.tick_claim_gc import (
    DELIVER_3_GOAL_INDEX,
    TickClaimGCConfig,
    checkpoint_files_present,
    command_deliver_3,
    initialize_tick_claim_gc,
)


def test_checkpoint_completion_requires_the_state_file(tmp_path):
    assert checkpoint_files_present(tmp_path) is False
    (tmp_path / "metadata.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    assert checkpoint_files_present(tmp_path) is False
    (tmp_path / "state.msgpack").write_bytes(b"")
    assert checkpoint_files_present(tmp_path) is False
    (tmp_path / "state.msgpack").write_bytes(b"state")
    assert checkpoint_files_present(tmp_path) is True


def test_deliver_3_command_keeps_parameters():
    config = TickClaimGCConfig(
        num_envs=2,
        num_steps=2,
        num_updates=1,
        minibatch_size=2,
        hidden_size=8,
        goal_mode="workshop12",
    )
    _network, runner = initialize_tick_claim_gc(config)
    before = jax.tree.map(lambda value: value, runner.train_state.params)
    commanded = command_deliver_3(runner)
    assert int(commanded.current_goal[0]) == DELIVER_3_GOAL_INDEX
    assert int(commanded.current_goal[1]) == DELIVER_3_GOAL_INDEX
    assert jax.tree_util.tree_all(
        jax.tree.map(
            lambda left, right: bool(jax.numpy.array_equal(left, right)),
            before,
            commanded.train_state.params,
        )
    )
