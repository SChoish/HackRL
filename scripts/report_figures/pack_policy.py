"""One PACK-RESTORE episode of a saved Dual policy, on both kernels."""

from __future__ import annotations

import json
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np

from hackrl.dual_leo import init_dual_leo_teacher, load_dual_checkpoint
from hackrl.pack_restore import (
    PackRestoreAction,
    PackRestorePhase,
    PackRestoreSplit,
    PackRestoreStart,
    PackRestoreVariant,
    make_pack_restore_state,
    pack_restore_goal_vector,
    pack_restore_world_done,
)
from hackrl.pack_restore_gc import (
    DELIVER_3_GOAL_INDEX,
    NUM_ACTIONS,
    NUM_GOALS,
    PackRestoreGCActorCritic,
    _batch_inputs,
    config_from_pack_restore_gc_payload,
    initialize_pack_restore_gc,
    pack_restore_gc_inputs,
)
from hackrl.pack_restore import observe_pack_restore, pack_restore_step

from report_figures.render import draw_pair
from report_figures.storyboards import _pack_view

from report_figures.common import ROOT
CHECKPOINT = (
    ROOT
    / "runs/dual_leo_compare_v1/pack/dual/mutant/seed20/checkpoints/adapt_4096"
)
OUT = ROOT / "runs/figures_report_v1/policy/pack_dual_seed20"
CAPTION = (
    "Same Dual policy, seed 20, adapt_4096, run separately on each kernel. "
    "Actions may diverge. Validation natural reset, layout 0, loaded phase. "
    "One episode, not the five-seed table. Mode excess delivery for seeds 20-24 is 1, 0.5, 1, 0, 0.5."
)


def _load_params():
    recorded = json.loads((CHECKPOINT / "config.json").read_text())
    config = config_from_pack_restore_gc_payload(recorded)
    _network, template = initialize_pack_restore_gc(config)
    inputs = _batch_inputs(template.env_state, template.current_goal)
    _, leo_template, _ = init_dual_leo_teacher(config, inputs[0], inputs[1], NUM_GOALS, NUM_ACTIONS)
    runner, _leo = load_dual_checkpoint(CHECKPOINT, template, leo_template)
    network = PackRestoreGCActorCritic(hidden_size=config.policy_hidden_size)
    return network, runner.train_state.params


def _rollout(network, params, state, variant, apply):
    goal = jnp.asarray(DELIVER_3_GOAL_INDEX, dtype=jnp.int32)
    frames = []
    for _step in range(128):
        if bool(pack_restore_goal_vector(state)[DELIVER_3_GOAL_INDEX]) or bool(pack_restore_world_done(state)):
            break
        observation = observe_pack_restore(state)
        model_inputs = pack_restore_gc_inputs(observation, goal)
        policy, _value = apply(
            params,
            model_inputs[0][None],
            model_inputs[1][None],
            model_inputs[2][None],
        )
        action = int(np.argmax(np.asarray(policy.logits[0])))
        nxt = pack_restore_step(state, action, variant)
        frames.append((nxt, PackRestoreAction(action).name, state))
        state = nxt
    return frames, state


def main():
    print("loading", CHECKPOINT, flush=True)
    network, params = _load_params()
    apply = jax.jit(lambda p, m, n, g: network.apply(p, m, n, g))
    start = make_pack_restore_state(
        0,
        int(PackRestorePhase.LOADED),
        split=PackRestoreSplit.VALIDATION,
        start=PackRestoreStart.NATURAL,
        source_growth_period=8,
    )
    # The reported mode episodes use the fixed evaluation order. Layout 0 loaded
    # is state index 16. Re-create that start directly.
    fixed_frames, fixed_final = _rollout(network, params, start, PackRestoreVariant.FIXED, apply)
    mutant_frames, mutant_final = _rollout(network, params, start, PackRestoreVariant.MUTANT, apply)
    print(
        "fixed",
        len(fixed_frames),
        int(fixed_final.delivered_total),
        "mutant",
        len(mutant_frames),
        int(mutant_final.delivered_total),
        flush=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    length = max(len(fixed_frames), len(mutant_frames))
    pngs = []
    first_action = None
    first_grain = None
    for index in range(length):
        left_state, left_action, left_before = fixed_frames[min(index, len(fixed_frames) - 1)]
        right_state, right_action, right_before = mutant_frames[min(index, len(mutant_frames) - 1)]
        left = dict(_pack_view(left_state, left_action, left_before, right_state), kernel="fixed")
        right = dict(_pack_view(right_state, right_action, right_before, left_state), kernel="mutant")
        if index >= len(fixed_frames):
            left["action"] = "held"
        if index >= len(mutant_frames):
            right["action"] = "held"
        grain_differs = left["resources"][0] != right["resources"][0]
        action_differs = left["action"] != right["action"] or left["player"] != right["player"]
        left["diverges"] = grain_differs or action_differs
        right["diverges"] = grain_differs or action_differs
        path = OUT / f"frame_{index:03d}.png"
        draw_pair(
            left,
            right,
            path,
            f"PACK-RESTORE Dual seed 20 adapt_4096  step {index + 1}",
            CAPTION,
        )
        pngs.append(path)
        if grain_differs and first_grain is None:
            first_grain = path
        if action_differs and first_action is None:
            first_action = path
    if first_grain is not None:
        (OUT / "first_grain_difference.png").write_bytes(first_grain.read_bytes())
    if first_action is not None:
        (OUT / "first_action_difference.png").write_bytes(first_action.read_bytes())
    (OUT / "final.png").write_bytes(pngs[-1].read_bytes())
    gif = OUT / "pair.gif"
    imageio.mimsave(gif, [imageio.imread(path) for path in pngs], duration=700)
    print("wrote", gif, "frames", len(pngs), flush=True)


if __name__ == "__main__":
    main()
