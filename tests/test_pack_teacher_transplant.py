import jax
import numpy as np
import pytest

from hackrl.pack_restore_gc import _evaluation_states
from hackrl.pack_restore import PackRestoreSplit


@pytest.mark.parametrize('repeats', [1, 2])
def test_natural_only_is_exact_historical_natural_subset(repeats):
    old = _evaluation_states(PackRestoreSplit.VALIDATION, repeats, 8)
    new = _evaluation_states(PackRestoreSplit.VALIDATION, repeats, 8, natural_only=True)
    for left, right in zip(jax.tree.leaves(old), jax.tree.leaves(new), strict=True):
        assert left.shape[0] == 64 * repeats
        assert right.shape[0] == 32 * repeats
        np.testing.assert_array_equal(left[:32*repeats], right)


def test_manifest_cohort_and_closed_bc_gate():
    import json
    from pathlib import Path
    spec = json.loads((Path(__file__).resolve().parents[1] /
                      'docs/manifests/pack_teacher_transplant_v1.json').read_text())
    assert spec['seeds'] == list(range(30,40))
    assert len(spec['seeds']) * len(spec['teacher_origins']) * 2 * 32 == 1280
    assert spec['evaluation']['total_episodes'] == 1280
    assert spec['student_bc']['common_data'] is None
    assert not spec['student_bc']['evaluation_data_may_enter_bc']


def test_natural_only_rollout_matches_historical_metrics():
    import distrax
    import jax.numpy as jnp
    from hackrl.pack_restore_gc import evaluate_pack_restore_gc_frozen
    from hackrl.pack_restore import PackRestoreAction

    class Noop:
        def apply(self, params, maps, numeric, goals):
            logits = jnp.zeros((maps.shape[0], len(PackRestoreAction)))
            logits = logits.at[:, int(PackRestoreAction.NOOP)].set(1)
            return distrax.Categorical(logits=logits), jnp.zeros(maps.shape[0])

    policy = Noop()
    kwargs = dict(variant='fixed', stochastic=False, repeats_per_state=1,
                  seed_base=20000, learner_seed=30, record_episodes=True)
    old = evaluate_pack_restore_gc_frozen(policy, {}, **kwargs)
    new = evaluate_pack_restore_gc_frozen(policy, {}, natural_only=True, **kwargs)
    assert new['natural_reset'] == old['natural_reset']
    assert new['overall'] == new['natural_reset']
    assert 'common_setup' not in new
    assert new['episode_records'] == old['episode_records'][:32]
