"""One-step real-action history with independent actor/critic and clock masks."""
from dataclasses import fields
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from hackrl import spatial_wall_pass as env
from hackrl import spatial_wall_pass_gc as gc
from hackrl.dual_leo import DualLeoQ, init_dual_leo_teacher, make_dual_leo_update
from hackrl.spatial_random_delay import DelayRunner, attach_delay, make_delay_outcome
from hackrl.spatial_wall_pass_reliability import base_runner

CONDITIONS = {'A': (True, True), 'B': (False, True), 'C': (True, False), 'D': (False, False)}
NUMERIC = len(env.MODEL_FEATURE_NAMES)
TIME = env.MODEL_FEATURE_NAMES.index('remaining_world_fraction')
TIME_INDICES = (TIME, NUMERIC + TIME)


@struct.dataclass
class History:
    maps: jax.Array
    numeric: jax.Array
    action: jax.Array
    valid: jax.Array


@struct.dataclass
class HistoryRunner(gc.SpatialWallPassGCRunnerState):
    history: History


@struct.dataclass
class HistoryDelayRunner(DelayRunner):
    history: History


def empty_history(state):
    maps, numeric, _ = gc._batch_inputs(state, jnp.zeros(state.tick.shape, jnp.int32))
    return History(jnp.zeros_like(maps), jnp.zeros_like(numeric),
                   jnp.zeros_like(state.tick), jnp.zeros_like(state.tick, dtype=bool))


def attach_history(runner):
    cls = HistoryDelayRunner if isinstance(runner, DelayRunner) else HistoryRunner
    values = {f.name: getattr(runner, f.name) for f in fields(type(runner)) if f.name != 'history'}
    return cls(**values, history=empty_history(runner.env_state))


def pack_inputs(state, goals, history):
    maps, numeric, goal = gc._batch_inputs(state, goals)
    valid = history.valid
    previous_maps = jnp.where(valid[:, None, None, None], history.maps, 0.)
    previous_numeric = jnp.where(valid[:, None], history.numeric, 0.)
    previous_action = jax.nn.one_hot(history.action, gc.NUM_ACTIONS) * valid[:, None]
    return (jnp.concatenate((maps, previous_maps), axis=-1),
            jnp.concatenate((numeric, previous_numeric, previous_action, valid[:, None]), axis=-1), goal)


def runner_inputs(runner):
    return pack_inputs(runner.env_state, runner.current_goal, runner.history)


def mask_time(numeric, enabled):
    """Runs on raw features, before any normalization/encoding."""
    if enabled:
        return numeric
    return numeric.at[:, jnp.asarray(TIME_INDICES)].set(0.)


def advance_history(history, state, actions, selected, world_done):
    maps, numeric, _ = gc._batch_inputs(state, jnp.zeros_like(actions))
    observed = History(maps, numeric, actions, jnp.ones_like(selected, dtype=bool))
    updated = gc._tree_where(selected, observed, history)
    # Last old-world action and forced reset waits are never initial history.
    return gc._tree_where(world_done, jax.tree.map(jnp.zeros_like, history), updated)


def history_outcome(*, adaptation=False, probability=.1, maximum=8):
    original = make_delay_outcome(probability, maximum) if adaptation else gc.spatial_wall_pass_outcome
    def outcome(runner, actions, config):
        before = runner.env_state
        result = original(runner, actions, config)
        out, done, valid, reward, terminals, world_done, goal_done, observed = result
        history = advance_history(runner.history, before, actions, valid, world_done)
        return out.replace(history=history), done, valid, reward, terminals, world_done, goal_done, observed
    return outcome


class HistoryEncoder(nn.Module):
    hidden_size: int = 512

    @nn.compact
    def __call__(self, maps, numeric, goal):
        spatial = nn.Conv(32, (3, 3), padding='SAME',
                          kernel_init=nn.initializers.lecun_normal(), bias_init=constant(0.), name='map_conv')(maps)
        spatial = nn.relu(spatial).reshape((spatial.shape[0], -1))
        x = jnp.concatenate((spatial, numeric, goal), axis=-1)
        return nn.relu(nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)),
                               bias_init=constant(0.), name='dense')(x))


class HistoryActorCritic(nn.Module):
    actor_time: bool = True
    hidden_size: int = 512
    action_dim: int = gc.NUM_ACTIONS

    @nn.compact
    def __call__(self, map_channels, numeric_features, goal_one_hot):
        actor = HistoryEncoder(self.hidden_size, name='actor_encoder')(
            map_channels, mask_time(numeric_features, self.actor_time), goal_one_hot)
        for i in range(2):
            actor = nn.relu(nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)),
                                    bias_init=constant(0.), name=f'actor_hidden_{i}')(actor))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(.01),
                          bias_init=constant(0.), name='actor_output')(actor)
        critic = HistoryEncoder(self.hidden_size, name='critic_encoder')(
            map_channels, numeric_features, goal_one_hot)
        for i in range(4):
            critic = nn.relu(nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)),
                                     bias_init=constant(0.), name=f'critic_hidden_{i}')(critic))
        value = nn.Dense(1, kernel_init=orthogonal(1.), bias_init=constant(0.), name='critic_output')(critic)
        return distrax.Categorical(logits=logits), jax.nn.sigmoid(value.squeeze(-1))


class HistoryTeacher(nn.Module):
    teacher_time: bool = True
    num_goals: int = gc.NUM_GOALS
    num_actions: int = gc.NUM_ACTIONS
    hidden_size: int = 512

    @nn.compact
    def __call__(self, map_channels, numeric_features, *, train):
        # Critically, BatchRenorm in DualLeoQ sees only already-masked inputs.
        numeric_features = mask_time(numeric_features, self.teacher_time)
        return DualLeoQ(self.num_goals, self.num_actions, self.hidden_size, name='q')(
            map_channels, numeric_features, train=train)


def initialize(config, condition):
    actor_time, teacher_time = CONDITIONS[condition]
    _, original = gc.initialize_spatial_wall_pass_gc(config)
    runner = attach_history(original)
    model = HistoryActorCritic(actor_time, config.policy_hidden_size)
    inputs = runner_inputs(runner)
    init_key = jax.random.split(jax.random.PRNGKey(config.seed), 4)[0]
    params = model.init(init_key, *(x[:1] for x in inputs))
    runner = runner.replace(train_state=TrainState.create(
        apply_fn=model.apply, params=params, tx=gc._optimizer(config)))
    teacher = HistoryTeacher(teacher_time=teacher_time, hidden_size=config.teacher_hidden_size)
    teacher, leo, mb = init_dual_leo_teacher(config, inputs[0], inputs[1], gc.NUM_GOALS,
                                           gc.NUM_ACTIONS, network=teacher)
    return model, runner, teacher, leo, mb


def adaptation_start(runner, seed):
    # Requires the original full normal state at a world reset boundary.
    commanded = gc.command_delivery(base_runner(runner))
    return attach_history(attach_delay(commanded, seed, .1, 8))


def make_update(model, teacher, config, mb, *, adaptation=False):
    return make_dual_leo_update(model, teacher, config, history_outcome(adaptation=adaptation),
                               gc._batch_inputs, mb, runner_inputs=runner_inputs)
