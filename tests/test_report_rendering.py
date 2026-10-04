"""Renderer correctness against kernel geometry and deliberately unequal metrics."""
from pathlib import Path
import sys
from types import SimpleNamespace
import json

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("imageio")
pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_figures.common import frame_status, metadata_lines, state_at, visible_crop, world_geometry
from report_figures import plots, render_pack_pixel, render_tick_craft_pixel


def _kernel(initial):
    return {"initial": initial, "steps": [], "summary": {"success": False}}


def test_pack_geometry_preserves_actual_validation_obstacle():
    from hackrl.pack_restore import make_pack_restore_state, PackRestoreSplit
    state = make_pack_restore_state(0, 1, split=PackRestoreSplit.VALIDATION)
    initial = {"player_position": np.asarray(state.player_position).tolist()}
    trace = {"geometry": {"map_size": 16, "walkable": np.argwhere(state.walkable).tolist()},
             "kernels": {"fixed": _kernel(initial), "mutant": _kernel(initial)}}
    floor, bbox = world_geometry(trace, "pack")
    # Use the engine, not a duplicate expected rectangle, as reference.
    for r in range(bbox[0], bbox[1] + 1):
        for c in range(bbox[2], bbox[3] + 1):
            expected = 0 <= r < 16 and 0 <= c < 16 and bool(state.walkable[r, c])
            assert ((r, c) in floor) == expected
    from hackrl.pack_restore import _extra_walls
    from hackrl.tick_claim import transform_positions
    wall = tuple(np.asarray(transform_positions(_extra_walls(PackRestoreSplit.VALIDATION), 0, PackRestoreSplit.VALIDATION))[0])
    assert wall not in floor


def test_craft_crop_edge_is_open_floor_and_preserves_objects():
    geometry = {"map_size": 16, "source": [8, 8], "grid": [9, 9], "delivery": [10, 9]}
    trace = {"geometry": geometry, "kernels": {"fixed": _kernel({"player_position": [10, 8]})}}
    floor, bbox = world_geometry(trace, "craft")
    assert (bbox[0], bbox[2]) in floor  # camera border must not become a wall
    for name in ("source", "grid", "delivery"):
        assert tuple(geometry[name]) not in floor


def test_observation_intersection_moves_with_player():
    observation = {"map_tiles": [[0] * 9 for _ in range(7)]}
    assert visible_crop({"player_position": [10, 8]}, observation, (6, 12, 6, 12)) == (7, 12, 6, 12)
    assert visible_crop({"player_position": [8, 8]}, observation, (6, 12, 6, 12)) == (6, 11, 6, 12)


def test_empty_and_failed_rollouts_are_not_labeled_success():
    kernel = _kernel({"tick": 128})
    state, step, held = state_at(kernel, 2)
    assert step is None and held
    assert frame_status(kernel, state, held) == "held: end 128"
    kernel["summary"]["success"] = True
    assert "success" in frame_status(kernel, state, held)


def test_no_exploit_trace_can_render_without_inventing_key_event():
    trace = {"kernels": {"mutant": {"steps": []}}}
    assert render_tick_craft_pixel._key_steps(trace, "tick", 0) == (("initial", 0), ("final", 0))
    assert render_tick_craft_pixel._key_steps(trace, "craft", 0) == (("initial", 0), ("final", 0))


def test_metadata_discloses_selection_and_mode_without_population_claim():
    trace = {"policy": {"method": "Dual", "seed": 21, "adaptation_updates": 4096},
             "start": {"family": "natural_reset", "phase": "ripe"},
             "selection": {"rule": "max length gain among successful exploit pairs"}}
    caption = " ".join(metadata_lines(trace))
    assert "mode" in caption and "max length gain" in caption
    assert "Synthetic" in caption and "not a mean" in caption
    assert "2/5" not in caption and "sample" not in caption


def test_conservation_badge_uses_recorded_amount():
    assert render_pack_pixel._event_label({"events": {"conservation_violation": True, "physical_created": 2}}) == "CONSERVATION +2"


def test_edv_keeps_comparators_distinct(monkeypatch):
    def block(success, exploit, value):
        return {"mode": {"natural_reset": {"success_rate": success, "exploit_rate": exploit, "mean_discounted_return": value}}}
    mutant = {"fixed": block(.4, 0, .8), "mutant": block(.8, .2, .7)}
    continued = {"fixed": block(1., 0, .9), "mutant": block(1., .5, .9)}
    monkeypatch.setattr(plots, "_curve", lambda root, env, method, variant, seed, update: mutant if variant == "mutant" else continued)
    groups = plots._edv_groups("pack", (("Dual", "dual", Path(".")),), "mode", None)
    np.testing.assert_allclose(groups["E"][0][1], [.4] * 5)
    np.testing.assert_allclose(groups["D"][0][1], [-.3] * 5)
    np.testing.assert_allclose(groups["V"][0][1], [-.1] * 5)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plots._panel(ax, groups["D"], "U", "signed contrast")
    assert ax.get_ylim()[0] < -.3
    plt.close(fig)


@pytest.mark.parametrize("environment", ["pack", "tick", "craft"])
def test_recorded_kernel_actions_render_both_modes(environment, monkeypatch, tmp_path):
    """Actual kernel states, scripted actions; not a trained-policy result."""
    import jax.numpy as jnp
    if environment == "pack":
        from report_figures import record_pack_trace as recorder
        from hackrl.pack_restore import PackRestoreAction as A, PackRestoreVariant as V, make_pack_restore_state, PackRestoreStart
        initial = make_pack_restore_state(0, 1, start=PackRestoreStart.PATH_CHECK)
        actions = [A.MAKE_RECORD, A.PACK_STORAGE, A.REBUILD_EMPTY, A.WITHDRAW_ONE, A.RIGHT, A.DO, A.DO, A.DOWN, A.DELIVER]
    elif environment == "tick":
        from report_figures import record_tick_trace as recorder
        from hackrl.tick_claim import TickClaimAction as A, TickClaimVariant as V, make_tick_claim_state, TickClaimStart, tick_claim_step, tick_claim_setup_prefix, GROWTH_WAIT_TICKS
        initial = make_tick_claim_state(0, 0, start=TickClaimStart.NATURAL)
        for action in tick_claim_setup_prefix(0):
            initial = tick_claim_step(initial, action, V.FIXED)
        actions = [A.ARM_HARVEST] + [A.NOOP] * (int(initial.reservation_delay) - 1) + [A.DO] + [A.NOOP] * GROWTH_WAIT_TICKS + [A.DO, A.DOWN, A.DELIVER]
    else:
        from report_figures import record_craft_trace as recorder
        from hackrl.craft_remain_env import CraftRemainVariant as V, make_craft_remain_state, REFERENCE_ACTION_INDEX, SOURCE, GRID, DELIVERY
        initial = make_craft_remain_state()
        payload = json.loads((Path(__file__).parents[1] / "docs/manifests/craft_remain_v1_replay.json").read_text())
        actions = [REFERENCE_ACTION_INDEX[name] for name in payload["mutant_shortest_actions"]]
    kernels = {}
    # The recorder's policy interface is scripted only for this kernel/render test.
    # Bypass JIT for the Python action iterator; environment transitions are real.
    monkeypatch.setattr(recorder.jax, "jit", lambda function: function)
    for name, variant in (("fixed", V.FIXED), ("mutant", V.MUTANT)):
        iterator = iter(actions)
        def apply(*args):
            action = int(next(iterator))
            logits = jnp.full((1, recorder.NUM_ACTIONS), -20.).at[0, action].set(20.)
            return SimpleNamespace(logits=logits), jnp.zeros(1)
        policy = SimpleNamespace(apply=apply) if environment == "pack" else apply
        kernels[name] = recorder._record_kernel(policy, None, initial, variant, max_steps=len(actions))
    geometry = {"map_size": 16}
    if environment == "craft":
        geometry.update(source=np.asarray(SOURCE).tolist(), grid=np.asarray(GRID).tolist(), delivery=np.asarray(DELIVERY).tolist())
    else:
        geometry["walkable"] = np.argwhere(initial.walkable).tolist()
    trace = {"geometry": geometry, "kernels": kernels,
             "policy": {"method": "SCRIPTED CHECK", "seed": 0, "adaptation_updates": 0},
             "start": {"family": "path_check", "phase": "ripe/loaded"},
             "selection": {"rule": "verified action sequence; not a learned policy"},
             "provenance": {"comparison": "same_action_sequence", "action_selection": "scripted"}}
    final = max(len(k["steps"]) for k in kernels.values())
    for analysis in (False, True):
        for index in (0, final, final + 1):
            image = (render_pack_pixel._render(trace, index, analysis) if environment == "pack"
                     else render_tick_craft_pixel._render(trace, environment, index, analysis))
            assert image.size[0] == 1920
            image.save(tmp_path / f"{environment}_{index}_{analysis}.png")
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    assert kernels["mutant"]["summary"]["success"]
    assert not kernels["fixed"]["summary"]["conservation_violation_steps"]
