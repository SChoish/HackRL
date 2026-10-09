import importlib.util
import json
from pathlib import Path
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from hackrl.spatial_wall_pass_gc import evaluate_spatial_wall_pass_gc_frozen

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("transfer", ROOT / "scripts/evaluate_spatial_wall_pass_transfer.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def test_new_geometry_matches_kernel_and_frozen_distances():
    contract = json.loads(transfer.MANIFEST.read_text())
    reports = transfer.audit_geometry(contract)
    assert [(r["fixed"]["shortest_length"], r["mutant"]["shortest_length"]) for r in reports] == [
        (10,6), (10,6), (13,11), (10,10), (4,4), (4,4)]
    assert sum(r[k]["independently_checked_transitions"] for r in reports for k in ("fixed","mutant")) == 15300
    for r in reports[-2:]:
        assert r["mutant"]["with_wall_pass"]["length"] == 8


class NoopPolicy:
    def apply(self, params, maps, features, goals):
        logits = jnp.full((maps.shape[0],23), -1e6).at[:,0].set(0.)
        return distrax.Categorical(logits=logits), jnp.zeros(maps.shape[0])


def test_custom_initial_states_preserve_episode_mapping_and_immutability():
    contract = json.loads(transfer.MANIFEST.read_text())
    states = [transfer.make_state(contract,f,i) for i,f in enumerate(contract["fixtures"][:2])]
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    before = serialization.to_bytes(batch)
    result = evaluate_spatial_wall_pass_gc_frozen(
        NoopPolicy(), {}, variant="mutant", stochastic=True,
        repeats_per_state=3, seed_base=61000,learner_seed=132,
        record_episodes=True,horizon=4,initial_states=batch)
    assert result["episodes"] == 6
    assert result["success_rate"] == 0
    assert result["mean_length"] == 4
    assert [(e["layout_index"],e["repeat"]) for e in result["episode_records"]] == [
        (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    assert before == serialization.to_bytes(batch)
