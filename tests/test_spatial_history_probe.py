import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hackrl import spatial_architecture as a
from hackrl import spatial_history_probe as p
from hackrl import spatial_wall_pass as env


@pytest.fixture(scope='module')
def panel():
    return jax.jit(p.make_panel)()


def test_full_state_and_current_observation_match_all64(panel):
    rows = p.validate_panel(panel)
    assert len(rows) == 64 and all(r['included'] for r in rows)
    for x, y in zip(jax.tree.leaves(panel['blocked']), jax.tree.leaves(panel['waited'])):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(panel['blocked'].tick, panel['delay'] + 2)


def test_replay_is_legal_and_waits_are_not_history(panel):
    for i in (0, 15, 32, 63):
        state = env.make_spatial_wall_pass_state(panel['layout'][i], split='validation')
        for _ in range(int(panel['delay'][i])):
            state = env.spatial_wall_pass_step(state, 0, 'fixed')
        state = env.spatial_wall_pass_step(state, panel['move'][i], 'fixed')
        for x, y in zip(jax.tree.leaves(state), jax.tree.leaves(panel['prefix'])):
            np.testing.assert_array_equal(x, y[i])
    assert np.all(np.asarray(panel['prefix_history'].valid).sum(-1) == 1)
    assert np.all(np.asarray(panel['blocked_history'].valid).sum(-1) == 2)
    np.testing.assert_array_equal(panel['blocked_history'].maps, panel['waited_history'].maps)
    np.testing.assert_array_equal(panel['blocked_history'].numeric, panel['waited_history'].numeric)
    np.testing.assert_array_equal(panel['blocked_history'].action[:, -1], panel['dash'])
    np.testing.assert_array_equal(panel['waited_history'].action[:, -1], 0)


def test_mismatch_is_excluded_without_repair(panel):
    changed = {**panel, 'waited': panel['waited'].replace(tick=panel['waited'].tick.at[0].add(1))}
    rows = p.validate_panel(changed)
    assert not rows[0]['included'] and 'state.tick' in rows[0]['exclusions']
    assert sum(r['included'] for r in rows) == 63
    assert int(changed['waited'].tick[0]) == 3


def test_current_actor_negative_control_with_real_history(panel):
    model = a.ArchitectureActorCritic('current_mlp', critic_width=8, token_width=8, actor_width_override=8)
    state, history = panel['blocked'], panel['blocked_history']
    inputs = a.pack_inputs(state, jnp.full(64, 11, jnp.int32), history)
    params = model.init(jax.random.PRNGKey(8), *inputs)
    query = p.make_queries(model)
    np.testing.assert_array_equal(query(params, state, history),
                                  query(params, panel['waited'], panel['waited_history']))


class NoopPolicy:
    def apply(self, params, maps, numeric, goal):
        del params, numeric, goal
        logits = jnp.full((maps.shape[0], 23), -100.).at[:, 0].set(0.)
        return distrax.Categorical(logits=logits), jnp.zeros(maps.shape[0])


def test_continuation_horizon_not_restarted_and_exclusion_mask(panel):
    valid = jnp.ones(64, bool).at[0].set(False)
    trace = jax.device_get(p.make_continuation(NoopPolicy())({}, panel['blocked'], panel['blocked_history'], valid))
    np.testing.assert_array_equal(trace['active'].sum(0), np.where(valid, 64 - panel['blocked'].tick, 0))
    rows = p.validate_panel(panel)
    rows[0]['included'] = False
    records = p.continuation_records(trace, rows, 'blocked')
    assert len(records) == 63
    assert all(r['episode_length'] == 64 and not r['success'] and r['discounted_return'] == 0 for r in records)
    assert all(r['control_length'] == 64 - r['delay'] for r in records)


def test_absolute_and_remaining_return_are_distinct():
    trace = dict(active=np.array([[True],[True]]), tick=np.array([[12],[13]]),
                 action=np.array([[0],[22]]), position=np.array([[[8,8]],[[8,8]]]),
                 next_position=np.array([[[8,8]],[[8,8]]]), blocked=np.array([[False],[False]]),
                 delivery=np.array([[False],[True]]))
    rows = [dict(included=True,panel_index=0,layout=0,delay=10,branch_tick=12,
                 branch_actions=[18,0],position=[8,8])]
    r = p.continuation_records(trace, rows, 'wait')[0]
    assert r['success'] and r['episode_length'] == 14 and r['control_length'] == 4
    assert r['continuation_length'] == 2
    assert r['discounted_return'] == env.DISCOUNT ** 13
    assert r['remaining_discounted_return'] == env.DISCOUNT


def test_natural_enrichment_is_block_then_success_not_direction_change():
    records = [dict(success=True), dict(success=False), dict(success=True)]
    trace = dict(active=np.ones((3,3),bool),dash=np.array([[True,True,False],[False]*3,[False]*3]),
                 blocked=np.array([[True,True,False],[False]*3,[False]*3]),
                 position=np.zeros((3,3,2),int),next_position=np.zeros((3,3,2),int))
    trace['next_position'][2,0,0] = 1
    out = p.enrich_natural(records,trace)
    assert [r['path_class'] for r in out] == ['blocked_then_success','blocked_then_failure','no_block_success']
    assert out[0]['ticks_to_movement_after_block'] == 2
    assert out[1]['ticks_to_movement_after_block'] is None
