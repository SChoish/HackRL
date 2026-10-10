"""PPO actor architectures with identical bounded history and current-only auxiliaries."""
from dataclasses import fields
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from hackrl import spatial_wall_pass_gc as gc
from hackrl import spatial_wall_pass as env
from hackrl.dual_leo import DualLeoQ, init_dual_leo_teacher, make_dual_leo_update
from hackrl.spatial_random_delay import DelayRunner, attach_delay, make_delay_outcome
from hackrl.spatial_wall_pass_reliability import base_runner
from hackrl.spatial_time_history import HistoryEncoder

ARCHITECTURES = ('current_mlp', 'history_mlp', 'gru', 'transformer')
WINDOW = 8
NUMERIC = len(env.MODEL_FEATURE_NAMES)
# Selected by parameter counts only, never by task reward.
ACTOR_WIDTHS = dict(current_mlp=1024, history_mlp=704, gru=576, transformer=128)


@struct.dataclass
class History:
    maps: jax.Array
    numeric: jax.Array
    action: jax.Array
    valid: jax.Array


@struct.dataclass
class WindowRunner(gc.SpatialWallPassGCRunnerState):
    history: History


@struct.dataclass
class WindowDelayRunner(DelayRunner):
    history: History


def empty_history(state):
    maps, numeric, _ = gc._batch_inputs(state, jnp.zeros_like(state.tick))
    return History(jnp.zeros((maps.shape[0], WINDOW-1, *maps.shape[1:]), maps.dtype),
                   jnp.zeros((numeric.shape[0], WINDOW-1, NUMERIC), numeric.dtype),
                   jnp.zeros((maps.shape[0], WINDOW-1), jnp.int32),
                   jnp.zeros((maps.shape[0], WINDOW-1), bool))


def attach_history(runner):
    cls = WindowDelayRunner if isinstance(runner, DelayRunner) else WindowRunner
    values = {f.name: getattr(runner, f.name) for f in fields(type(runner)) if f.name != 'history'}
    return cls(**values, history=empty_history(runner.env_state))


def pack_inputs(state, goals, history):
    maps, numeric, goal = gc._batch_inputs(state, goals)
    valid = jnp.concatenate((history.valid, jnp.ones((maps.shape[0], 1), bool)), axis=1)
    maps = jnp.concatenate((history.maps, maps[:, None]), axis=1)
    numeric = jnp.concatenate((history.numeric, numeric[:, None]), axis=1)
    # Completed (observation, action) pairs, followed by current observation.
    # The current action is NOT known at forward time and is always zero-coded.
    actions = jnp.concatenate((jax.nn.one_hot(history.action, gc.NUM_ACTIONS),
                               jnp.zeros((maps.shape[0], 1, gc.NUM_ACTIONS))), axis=1)
    numeric = jnp.concatenate((numeric, actions, valid[..., None]), axis=-1)
    return (jnp.where(valid[..., None, None, None], maps, 0.),
            jnp.where(valid[..., None], numeric, 0.), goal)


def runner_inputs(runner):
    return pack_inputs(runner.env_state, runner.current_goal, runner.history)


def advance_history(history, state, actions, selected, world_done):
    maps, numeric, _ = gc._batch_inputs(state, jnp.zeros_like(actions))
    append = lambda old, new: jnp.concatenate((old[:, 1:], new[:, None]), axis=1)
    observed = History(append(history.maps, maps), append(history.numeric, numeric),
                       append(history.action, actions), append(history.valid, jnp.ones_like(selected)))
    updated = gc._tree_where(selected, observed, history)
    return gc._tree_where(world_done, jax.tree.map(jnp.zeros_like, history), updated)


def history_outcome(*, adaptation=False, probability=.1, maximum=8):
    original = make_delay_outcome(probability, maximum) if adaptation else gc.spatial_wall_pass_outcome
    def outcome(runner, actions, config):
        result = original(runner, actions, config)
        out, done, valid, reward, terminals, world_done, goal_done, observed = result
        history = advance_history(runner.history, runner.env_state, actions, valid, world_done)
        return out.replace(history=history), done, valid, reward, terminals, world_done, goal_done, observed
    return outcome


class CausalBlock(nn.Module):
    width: int
    heads: int = 4

    @nn.compact
    def __call__(self, x, valid):
        mask = nn.combine_masks(nn.make_causal_mask(valid), nn.make_attention_mask(valid, valid))
        z = nn.LayerNorm(name='attention_norm')(x)
        z = nn.MultiHeadDotProductAttention(num_heads=self.heads, qkv_features=self.width,
                                            dropout_rate=0., name='attention')(z, z, mask=mask)
        x = x + z
        z = nn.LayerNorm(name='ff_norm')(x)
        z = nn.gelu(nn.Dense(4*self.width, name='ff_in')(z))
        x = x + nn.Dense(self.width, name='ff_out')(z)
        return jnp.where(valid[..., None], x, 0.)


class Actor(nn.Module):
    architecture: str
    token_width: int = 128
    width_override: int = 0
    transformer_layers: int = 6

    @nn.compact
    def __call__(self, maps, numeric, goal):
        if self.architecture not in ARCHITECTURES:
            raise ValueError(self.architecture)
        width = self.width_override or ACTOR_WIDTHS[self.architecture]
        if self.architecture == 'current_mlp':
            maps, numeric = maps[:, -1:], numeric[:, -1:]
            # Identical token shape, but no history validity or action information.
            numeric = numeric.at[..., NUMERIC:].set(0.)
            valid = jnp.ones(numeric.shape[:2], bool)
        else:
            valid = numeric[..., -1] > 0
        maps = jnp.where(valid[..., None, None, None], maps, 0.)
        numeric = jnp.where(valid[..., None], numeric, 0.)
        b, t = maps.shape[:2]
        spatial = nn.relu(nn.Conv(32, (3, 3), padding='SAME', name='map_conv')(
            maps.reshape((b*t, *maps.shape[2:])))).reshape((b, t, -1))
        goal_tokens = jnp.broadcast_to(goal[:, None], (b, t, goal.shape[-1]))
        tokens = nn.relu(nn.Dense(self.token_width, name='token_encoder')(
            jnp.concatenate((spatial, numeric, goal_tokens), axis=-1)))
        tokens = jnp.where(valid[..., None], tokens, 0.)
        if self.architecture in ('current_mlp', 'history_mlp'):
            x = tokens.reshape((b, -1))
            for i in range(2):
                x = nn.relu(nn.Dense(width, kernel_init=orthogonal(np.sqrt(2)), name=f'hidden_{i}')(x))
        elif self.architecture == 'gru':
            cell = nn.GRUCell(features=width, name='gru')
            x = jnp.zeros((b, width), tokens.dtype)
            for i in range(t):
                updated, _ = cell(x, tokens[:, i])
                x = jnp.where(valid[:, i, None], updated, x)
        else:
            if self.token_width != width:
                tokens = nn.Dense(width, name='projection')(tokens)
            position = self.param('position', nn.initializers.normal(.02), (WINDOW, width))
            x = jnp.where(valid[..., None], tokens + position[None], 0.)
            for i in range(self.transformer_layers):
                x = CausalBlock(width, name=f'block_{i}')(x, valid)
            x = nn.LayerNorm(name='output_norm')(x[:, -1])
        return nn.Dense(gc.NUM_ACTIONS, kernel_init=orthogonal(.01),
                        bias_init=constant(0.), name='output')(x)


class ArchitectureActorCritic(nn.Module):
    architecture: str
    critic_width: int = 512
    token_width: int = 128
    actor_width_override: int = 0
    transformer_layers: int = 6

    @nn.compact
    def __call__(self, maps, numeric, goal):
        logits = Actor(self.architecture, self.token_width, self.actor_width_override,
                       self.transformer_layers, name='actor')(maps, numeric, goal)
        # Current observation only; completely disjoint from the actor parameters.
        x = HistoryEncoder(self.critic_width, name='critic_encoder')(maps[:, -1], numeric[:, -1, :NUMERIC], goal)
        for i in range(4):
            x = nn.relu(nn.Dense(self.critic_width, kernel_init=orthogonal(np.sqrt(2)),
                                name=f'critic_hidden_{i}')(x))
        value = nn.Dense(1, kernel_init=orthogonal(1.), name='critic_output')(x)
        return distrax.Categorical(logits=logits), jax.nn.sigmoid(value.squeeze(-1))


class CurrentTeacher(nn.Module):
    num_goals: int = gc.NUM_GOALS
    num_actions: int = gc.NUM_ACTIONS
    hidden_size: int = 512

    @nn.compact
    def __call__(self, maps, numeric, *, train):
        return DualLeoQ(self.num_goals, self.num_actions, self.hidden_size, name='q')(
            maps[:, -1], numeric[:, -1, :NUMERIC], train=train)


def initialize(config, architecture, *, small_test=False):
    _, original = gc.initialize_spatial_wall_pass_gc(config)
    runner = attach_history(original)
    model = ArchitectureActorCritic(architecture, config.policy_hidden_size,
        token_width=8 if small_test else 128, actor_width_override=16 if small_test else 0,
        transformer_layers=1 if small_test else 6)
    inputs = runner_inputs(runner)
    key = jax.random.split(jax.random.PRNGKey(config.seed), 4)[0]
    params = model.init(key, *(x[:1] for x in inputs))
    runner = runner.replace(train_state=TrainState.create(apply_fn=model.apply, params=params, tx=gc._optimizer(config)))
    teacher = CurrentTeacher(hidden_size=config.teacher_hidden_size)
    teacher, leo, mb = init_dual_leo_teacher(config, inputs[0], inputs[1], gc.NUM_GOALS,
                                           gc.NUM_ACTIONS, network=teacher)
    return model, runner, teacher, leo, mb


def adaptation_start(runner, seed, probability):
    return attach_history(attach_delay(gc.command_delivery(base_runner(runner)), seed, probability, 8))


def make_update(model, teacher, config, mb, *, adaptation=False, probability=.1):
    return make_dual_leo_update(model, teacher, config,
        history_outcome(adaptation=adaptation, probability=probability), gc._batch_inputs, mb,
        runner_inputs=runner_inputs)
