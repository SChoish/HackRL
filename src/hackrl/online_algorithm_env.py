"""Shared TICK-CLAIM/PACK-RESTORE wiring for online value algorithms.

The fixture runners remain the single source of truth for reset, command
switching, terminal masks, and defect audit events.  This module replaces only
their learner and action-selection policy.  It defines no experiment candidate
and performs no execution on import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import distrax
import jax
import jax.numpy as jnp
from flax import struct

from hackrl import pack_restore_gc, tick_claim_gc
from hackrl.discrete_sac import (
    SDSACBatch,
    append_sd_sac_replay,
    categorical_terms,
    init_sd_sac,
    record_sd_sac_environment_steps,
)
from hackrl.online_value import (
    AllGoalTransition,
    DualOnlineQState,
    GoalQTransition,
    all_goal_q_apply,
    combine_dual_q,
    epsilon_at_phase_step,
    epsilon_greedy_actions,
    goal_q_apply,
    init_all_goal_q,
    init_goal_q,
    reset_phase_counter,
    select_goal_q,
    update_all_goal_q,
    update_dual_q,
    update_goal_q,
)


PQN = "GC-PQN"
LEO = "LEO"
DUAL = "Dual LEO(PQN)"
SD_SAC = "GC-SD-SAC"
ONLINE_VALUE_METHODS = (PQN, LEO, DUAL)


@dataclass(frozen=True)
class OnlineEnvironmentAdapter:
    name: str
    num_goals: int
    num_actions: int
    initialize: Callable
    batch_inputs: Callable
    step: Callable
    event_valid: Callable


@dataclass(frozen=True)
class OnlineValueNetworks:
    pqn: Any = None
    leo: Any = None


class OnlineTransitionPair(struct.PyTreeNode):
    pqn: GoalQTransition
    leo: AllGoalTransition


@dataclass(frozen=True)
class OnlineValueEvaluationNetwork:
    method: str
    networks: OnlineValueNetworks

    def apply(self, state, *model_inputs):
        q_values = _acting_q(self.method, self.networks, state, model_inputs)
        return distrax.Categorical(logits=q_values), jnp.zeros(
            q_values.shape[:-1], dtype=jnp.float32
        )


@dataclass(frozen=True)
class SDSACEvaluationNetwork:
    actor_network: Any

    def apply(self, state, *model_inputs):
        actor = state.actor
        logits = self.actor_network.apply(
            {"params": actor.params, "batch_stats": actor.batch_stats},
            *model_inputs,
            train=False,
        )
        return distrax.Categorical(logits=logits), jnp.zeros(
            logits.shape[:-1], dtype=jnp.float32
        )


def environment_adapter(name: str) -> OnlineEnvironmentAdapter:
    normalized = str(name).strip().lower().replace("_", "-")
    if normalized in {"tick", "tick-claim"}:
        return OnlineEnvironmentAdapter(
            name="TICK-CLAIM",
            num_goals=tick_claim_gc.NUM_GOALS,
            num_actions=tick_claim_gc.NUM_ACTIONS,
            initialize=tick_claim_gc.initialize_tick_claim_gc,
            batch_inputs=tick_claim_gc._batch_inputs,
            step=tick_claim_gc.step_tick_claim_gc_workers,
            event_valid=lambda event: event.valid_transition,
        )
    if normalized in {"pack", "pack-restore"}:
        return OnlineEnvironmentAdapter(
            name="PACK-RESTORE",
            num_goals=pack_restore_gc.NUM_GOALS,
            num_actions=pack_restore_gc.NUM_ACTIONS,
            initialize=pack_restore_gc.initialize_pack_restore_gc,
            batch_inputs=pack_restore_gc._batch_inputs,
            step=pack_restore_gc.step_pack_restore_gc_workers,
            event_valid=lambda event: event.valid,
        )
    raise ValueError(f"unknown online-algorithm environment: {name!r}")


def _done(event):
    return jnp.logical_or(event.goal_done, event.world_done)


def build_online_transitions(
    adapter: OnlineEnvironmentAdapter,
    model_inputs,
    action,
    event,
    next_inputs,
):
    """Translate one canonical fixture step into both online TD views."""

    valid = adapter.event_valid(event)
    pqn = GoalQTransition(
        map_channels=model_inputs[0],
        numeric_features=model_inputs[1],
        goal_one_hot=model_inputs[2],
        action=action,
        reward=event.reward,
        next_map_channels=next_inputs[0],
        next_numeric_features=next_inputs[1],
        next_goal_one_hot=next_inputs[2],
        done=_done(event),
        valid=valid,
    )
    leo = AllGoalTransition(
        map_channels=model_inputs[0],
        numeric_features=model_inputs[1],
        action=action,
        terminal_goals=event.terminal_goals,
        next_map_channels=next_inputs[0],
        next_numeric_features=next_inputs[1],
        world_done=event.world_done,
        valid=valid,
    )
    return OnlineTransitionPair(pqn=pqn, leo=leo)


def _flatten_rollout(transition):
    return jax.tree.map(
        lambda value: value.reshape((-1,) + value.shape[2:]), transition
    )


def _schedule_state(method, state):
    return state.pqn if method == DUAL else state


def rollout_phase_steps(runner_environment_steps, schedule_state):
    """Include physical transitions collected in the not-yet-updated rollout."""

    pending = runner_environment_steps - schedule_state.environment_steps
    return schedule_state.phase_steps + pending


def _acting_q(method, networks, state, model_inputs):
    if method == PQN:
        return goal_q_apply(networks.pqn, state, *model_inputs)
    if method == LEO:
        all_q = all_goal_q_apply(
            networks.leo, state, model_inputs[0], model_inputs[1]
        )
        return select_goal_q(all_q, jnp.argmax(model_inputs[2], axis=-1))
    if method == DUAL:
        pqn_q = goal_q_apply(networks.pqn, state.pqn, *model_inputs)
        all_q = all_goal_q_apply(
            networks.leo, state.leo, model_inputs[0], model_inputs[1]
        )
        leo_q = select_goal_q(all_q, jnp.argmax(model_inputs[2], axis=-1))
        return combine_dual_q(pqn_q, leo_q, leo_weight=0.3)
    raise ValueError(f"unsupported online value method: {method!r}")


def initialize_online_value_runner(
    adapter: OnlineEnvironmentAdapter,
    env_config,
    *,
    method: str,
    pqn_hidden_size: int,
    leo_hidden_size: int,
    learning_rate: float,
    max_grad_norm: float,
):
    """Initialize the canonical fixture state and replace its PPO learner."""

    if method not in ONLINE_VALUE_METHODS:
        raise ValueError(f"unsupported online value method: {method!r}")
    _, runner = adapter.initialize(env_config)
    inputs = adapter.batch_inputs(runner.env_state, runner.current_goal)
    next_rng, pqn_key, leo_key = jax.random.split(runner.rng, 3)
    networks = OnlineValueNetworks()
    if method in {PQN, DUAL}:
        pqn_network, pqn_state = init_goal_q(
            pqn_key,
            *inputs,
            num_actions=adapter.num_actions,
            hidden_size=pqn_hidden_size,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
        )
        networks = OnlineValueNetworks(pqn=pqn_network, leo=networks.leo)
    if method in {LEO, DUAL}:
        leo_network, leo_state = init_all_goal_q(
            leo_key,
            inputs[0],
            inputs[1],
            num_goals=adapter.num_goals,
            num_actions=adapter.num_actions,
            hidden_size=leo_hidden_size,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
        )
        networks = OnlineValueNetworks(pqn=networks.pqn, leo=leo_network)
    if method == PQN:
        learner = pqn_state
    elif method == LEO:
        learner = leo_state
    else:
        learner = DualOnlineQState(pqn=pqn_state, leo=leo_state)
    return networks, runner.replace(train_state=learner, rng=next_rng)


def make_online_value_update(
    adapter: OnlineEnvironmentAdapter,
    env_config,
    networks: OnlineValueNetworks,
    *,
    method: str,
    epsilon_start: float,
    epsilon_finish: float,
    epsilon_decay_transitions: int,
):
    """Build one frozen-rollout online update; no replay is retained."""

    if method not in ONLINE_VALUE_METHODS:
        raise ValueError(f"unsupported online value method: {method!r}")
    if not 0.0 <= epsilon_finish <= epsilon_start <= 1.0:
        raise ValueError("epsilon must satisfy 0 <= finish <= start <= 1")

    def rollout_step(runner, _):
        inputs = adapter.batch_inputs(runner.env_state, runner.current_goal)
        q_values = _acting_q(method, networks, runner.train_state, inputs)
        schedule = _schedule_state(method, runner.train_state)
        epsilon = epsilon_at_phase_step(
            rollout_phase_steps(runner.env_steps, schedule),
            start=epsilon_start,
            finish=epsilon_finish,
            decay_transitions=epsilon_decay_transitions,
        )
        next_rng, action_key = jax.random.split(runner.rng)
        action = epsilon_greedy_actions(action_key, q_values, epsilon)
        stepped_runner, event = adapter.step(
            runner.replace(rng=next_rng), action, env_config
        )
        next_inputs = adapter.batch_inputs(
            stepped_runner.env_state, stepped_runner.current_goal
        )
        transition = build_online_transitions(
            adapter, inputs, action, event, next_inputs
        )
        return stepped_runner, (transition, epsilon)

    def update(runner):
        runner, (trajectory, epsilon) = jax.lax.scan(
            rollout_step, runner, None, length=env_config.num_steps
        )
        flat = _flatten_rollout(trajectory)
        if method == PQN:
            learner, learning = update_goal_q(
                networks.pqn, runner.train_state, flat.pqn, gamma=env_config.gamma
            )
        elif method == LEO:
            learner, learning = update_all_goal_q(
                networks.leo, runner.train_state, flat.leo, gamma=env_config.gamma
            )
        else:
            learner, learning = update_dual_q(
                networks.pqn,
                networks.leo,
                runner.train_state,
                flat.pqn,
                flat.leo,
                gamma=env_config.gamma,
            )
        runner = runner.replace(
            train_state=learner, global_update=runner.global_update + 1
        )
        metrics = {
            "learning": learning,
            "epsilon_start": epsilon[0],
            "epsilon_end": epsilon[-1],
            "physical_transitions": jnp.asarray(
                env_config.batch_size, dtype=jnp.int32
            ),
            "valid_transitions": jnp.sum(flat.pqn.valid.astype(jnp.int32)),
            "retained_rollout_batches": jnp.asarray(0, dtype=jnp.int32),
        }
        return runner, metrics

    return update


def reset_online_value_adaptation(runner, *, method: str):
    """Restart only phase-local exploration after an immutable branch clone."""

    state = runner.train_state
    if method == DUAL:
        state = DualOnlineQState(
            pqn=reset_phase_counter(state.pqn),
            leo=reset_phase_counter(state.leo),
        )
    elif method in {PQN, LEO}:
        state = reset_phase_counter(state)
    else:
        raise ValueError(f"unsupported online value method: {method!r}")
    return runner.replace(train_state=state)


def clone_training_branch(tree):
    """Materialize an identical functional branch, including RNG and replay."""

    return jax.tree.map(lambda leaf: jnp.array(leaf, copy=True), tree)


def frozen_evaluation_binding(
    *, method: str, networks=None, state=None, actor_network=None, stochastic=False
):
    """Return a network/state pair accepted by the canonical frozen evaluators.

    Value methods expose only their registered epsilon-zero greedy view here.
    Treating raw Q values as categorical sampling logits would define a
    different, unregistered policy and is rejected.
    """

    if method in ONLINE_VALUE_METHODS:
        if stochastic:
            raise ValueError(
                "online Q stochastic evaluation needs a separately frozen "
                "epsilon-behavior contract"
            )
        if networks is None or state is None:
            raise ValueError("online value evaluation needs networks and state")
        return OnlineValueEvaluationNetwork(method, networks), state
    if method == SD_SAC:
        if actor_network is None or state is None:
            raise ValueError("SD-SAC evaluation needs actor_network and state")
        return SDSACEvaluationNetwork(actor_network), state
    raise ValueError(f"unsupported evaluation method: {method!r}")


def initialize_sd_sac_runner(
    adapter: OnlineEnvironmentAdapter,
    env_config,
    *,
    hidden_size: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
    temperature_learning_rate: float,
    initial_alpha: float,
    max_grad_norm: float,
):
    _, runner = adapter.initialize(env_config)
    inputs = adapter.batch_inputs(runner.env_state, runner.current_goal)
    next_rng, init_key = jax.random.split(runner.rng)
    actor, critic_1, critic_2, state = init_sd_sac(
        init_key,
        *inputs,
        num_actions=adapter.num_actions,
        hidden_size=hidden_size,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        temperature_learning_rate=temperature_learning_rate,
        initial_alpha=initial_alpha,
        max_grad_norm=max_grad_norm,
    )
    return (actor, critic_1, critic_2), runner.replace(
        train_state=state, rng=next_rng
    )


def make_sd_sac_collection(
    adapter: OnlineEnvironmentAdapter,
    env_config,
    actor_network,
):
    """Collect canonical transitions and append their behavior entropy."""

    def rollout_step(runner, _):
        inputs = adapter.batch_inputs(runner.env_state, runner.current_goal)
        actor = runner.train_state.actor
        logits = actor_network.apply(
            {"params": actor.params, "batch_stats": actor.batch_stats},
            *inputs,
            train=False,
        )
        _, _, entropy = categorical_terms(logits)
        next_rng, action_key = jax.random.split(runner.rng)
        action = jax.random.categorical(action_key, logits).astype(jnp.int32)
        stepped_runner, event = adapter.step(
            runner.replace(rng=next_rng), action, env_config
        )
        next_inputs = adapter.batch_inputs(
            stepped_runner.env_state, stepped_runner.current_goal
        )
        batch = SDSACBatch(
            map_channels=inputs[0],
            numeric_features=inputs[1],
            goal_one_hot=inputs[2],
            action=action,
            reward=event.reward,
            next_map_channels=next_inputs[0],
            next_numeric_features=next_inputs[1],
            next_goal_one_hot=next_inputs[2],
            done=_done(event),
            valid=adapter.event_valid(event),
            behavior_entropy=entropy,
        )
        return stepped_runner, batch

    def collect(runner, replay):
        runner, trajectory = jax.lax.scan(
            rollout_step, runner, None, length=env_config.num_steps
        )
        flat = _flatten_rollout(trajectory)
        replay = append_sd_sac_replay(replay, flat)
        state = record_sd_sac_environment_steps(
            runner.train_state, env_config.batch_size
        )
        runner = runner.replace(train_state=state)
        metrics = {
            "physical_transitions": jnp.asarray(
                env_config.batch_size, dtype=jnp.int32
            ),
            "valid_transitions": jnp.sum(flat.valid.astype(jnp.int32)),
            "replay_size": replay.size,
            "replay_total_inserted": replay.total_inserted,
        }
        return runner, replay, metrics

    return collect
