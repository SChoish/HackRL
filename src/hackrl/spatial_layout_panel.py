"""Policy-free layout generation and exact directed, unit-cost route search."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

SIZE = 16
SUPPORT = {(r, c) for r in range(6, 13) for c in range(6, 13)} - {(12, 12)}
FAMILIES = ('straight_partial', 'bent', 'staggered', 'doorway')
DELTAS = np.array(((0, -1), (0, 1), (-1, 0), (1, 0)), dtype=np.int32)


def fixture(walls, camp, start, item, direction=2, family='reference'):
    walls = set(map(tuple, walls))
    assert walls <= SUPPORT
    roles = tuple(map(tuple, (camp, start, item)))
    assert len(set(roles)) == 3 and all(x in SUPPORT - walls for x in roles)
    assert 1 <= direction <= 4
    floor = np.zeros((SIZE, SIZE), bool)
    barrier = np.zeros_like(floor)
    for pos in SUPPORT - walls:
        floor[pos] = True
    for pos in walls:
        barrier[pos] = True
    return dict(walkable=floor.tolist(), wall_mask=barrier.tolist(),
                camp=list(camp), start=list(start), item=list(item),
                direction=int(direction), family=family)


def generate_candidate(rng, index):
    family = FAMILIES[index % len(FAMILIES)]
    walls = set()
    if family == 'straight_partial':
        c, length = int(rng.integers(7, 12)), int(rng.integers(2, 7))
        begin = int(rng.integers(6, 14 - length))
        walls = {(r, c) for r in range(begin, begin + length)}
    elif family == 'bent':
        r, c = map(int, rng.integers(7, 12, 2))
        vertical, horizontal = map(int, rng.integers(2, 6, 2))
        dr, dc = rng.choice((-1, 1), 2)
        walls = {(r + int(dr) * k, c) for k in range(vertical)}
        walls |= {(r, c + int(dc) * k) for k in range(horizontal)}
    elif family == 'staggered':
        cols = sorted(map(int, rng.choice(np.arange(7, 12), int(rng.integers(2, 4)), replace=False)))
        flip = int(rng.integers(2))
        for k, c in enumerate(cols):
            length = int(rng.integers(3, 7))
            begin = 6 if (k + flip) % 2 == 0 else 13 - length
            walls |= {(r, c) for r in range(begin, begin + length)}
    else:
        c, door = int(rng.integers(7, 12)), int(rng.integers(6, 13))
        walls = {(r, c) for r in range(6, 13) if r != door}
        if rng.integers(2):
            r, door = int(rng.integers(7, 12)), int(rng.integers(6, 13))
            walls |= {(r, c) for c in range(6, 13) if c != door}
    if rng.integers(2):
        walls = {(c, r) for r, c in walls}
    walls &= SUPPORT
    free = sorted(SUPPORT - walls)
    role_ids = rng.choice(len(free), 3, replace=False)
    return fixture(walls, *(free[i] for i in role_ids),
                   direction=int(rng.integers(1, 5)), family=family)


def canonical_hash(f, *, include_direction=True):
    """D4 plus translations; includes complete support and all endpoint roles."""
    floor = np.argwhere(f['walkable'])
    wall = np.argwhere(f['wall_mask'])
    roles = np.asarray([f['camp'], f['start'], f['item']])
    direction = DELTAS[f['direction'] - 1]
    versions = []
    for reflect in (False, True):
        for rotate in range(4):
            matrix = np.array(((1, 0), (0, -1 if reflect else 1)), int)
            for _ in range(rotate):
                matrix = np.array(((0, 1), (-1, 0))) @ matrix
            fl, wl, ro = floor @ matrix.T, wall @ matrix.T, roles @ matrix.T
            minimum = np.concatenate((fl, wl)).min(axis=0)
            entry = [sorted(map(tuple, fl - minimum)), sorted(map(tuple, wl - minimum)),
                     list(map(tuple, ro - minimum))]
            if include_direction:
                entry.append(tuple(direction @ matrix.T))
            versions.append(json.dumps(entry, separators=(',', ':'), default=lambda x: int(x)))
    return hashlib.sha256(min(versions).encode()).hexdigest()


@dataclass
class RouteGraph:
    positions: np.ndarray
    nodes: np.ndarray
    targets: np.ndarray
    initial: int
    transition_fields: dict

    def paths(self):
        n = len(self.nodes)
        rows = np.repeat(np.arange(n), 23)
        cols = self.targets.ravel()
        keep = rows != cols
        graph = csr_matrix((np.ones(keep.sum(), np.int8), (rows[keep], cols[keep])), shape=(n+2, n+2))
        _, predecessors = breadth_first_order(graph, self.initial, directed=True, return_predecessors=True)
        paths = []
        for end in (n, n+1):
            if predecessors[end] < 0:
                paths.append(None)
                continue
            steps = []
            while end != self.initial:
                prev = int(predecessors[end])
                steps.append(int(np.flatnonzero(self.targets[prev] == end)[0]))
                end = prev
            paths.append(steps[::-1])
        return paths


def route_graph(f, mutant):
    """Independent vectorized model; full position/direction/task/cross state."""
    walkable, walls = np.asarray(f['walkable']), np.asarray(f['wall_mask'])
    positions = np.argwhere(walkable)
    nodes = np.array(list(np.ndindex(len(positions), 4, 2, 2)), dtype=np.int32)
    p, direction, carrying, crossed = nodes.T
    n = len(nodes)
    lookup = np.full((SIZE, SIZE), -1, np.int32)
    lookup[positions[:, 0], positions[:, 1]] = np.arange(len(positions))
    actions = np.arange(23)
    move = (1 <= actions) & (actions <= 4)
    dash = (17 <= actions) & (actions <= 20)
    movement = move | dash
    requested_direction = np.clip(np.where(dash, actions - 17, actions - 1), 0, 3)
    delta = DELTAS[requested_direction]
    source = positions[:, None, :]
    mid = source + delta[None]
    destination = source + delta[None] * np.where(dash, 2, 1)[None, :, None]
    inside = ((destination >= 0) & (destination < SIZE)).all(-1)
    mid_inside = ((mid >= 0) & (mid < SIZE)).all(-1)
    dst_safe, mid_safe = np.clip(destination, 0, SIZE-1), np.clip(mid, 0, SIZE-1)
    mid_wall = walls[mid_safe[..., 0], mid_safe[..., 1]]
    dst_ok = inside & walkable[dst_safe[..., 0], dst_safe[..., 1]] & (destination != f['camp']).any(-1)
    mid_ok = (mid_inside & (walkable[mid_safe[..., 0], mid_safe[..., 1]] | mid_wall)
              & (mid != f['camp']).any(-1) & (mutant | ~mid_wall))
    allowed = movement[None] & dst_ok & (~dash[None] | mid_ok)
    new_position = np.where(allowed, lookup[dst_safe[..., 0], dst_safe[..., 1]], np.arange(len(positions))[:, None])
    wall_pass = dash[None] & allowed & mid_wall
    next_p = new_position[p]
    next_d = np.where(movement[None], requested_direction[None], direction[:, None])
    next_cross = crossed[:, None] | wall_pass[p]
    at_item = (positions[next_p] == f['item']).all(-1)
    pickup = (actions[None] == 21) & at_item & ~carrying[:, None].astype(bool)
    next_carry = carrying[:, None] | pickup
    adjacent = np.abs(positions[next_p] - f['camp']).sum(-1) == 1
    delivered = (actions[None] == 22) & adjacent & next_carry.astype(bool)
    targets = (((next_p * 4 + next_d) * 2 + next_carry) * 2 + next_cross).astype(np.int32)
    targets = np.where(delivered, n + next_cross, targets).astype(np.int32)
    initial = ((int(lookup[tuple(f['start'])]) * 4 + f['direction'] - 1) * 2) * 2
    return RouteGraph(positions, nodes, targets, initial,
        dict(position=positions[next_p], direction=next_d+1, carrying=next_carry & ~delivered,
             crossed=next_cross.astype(bool), delivered=delivered, picked_up=pickup,
             wall_pass=wall_pass[p], dash_succeeded=allowed[p] & dash[None]))


def solve(f):
    # No-cross subgraph of mutant must coincide with fixed; checked separately.
    graph = route_graph(f, True)
    normal, crossing = graph.paths()
    return dict(L0=None if normal is None else len(normal),
                Lcross=None if crossing is None else len(crossing),
                A=None if normal is None or crossing is None else len(normal)-len(crossing),
                normal_actions=normal, crossing_actions=crossing)


def classify(result):
    l0, lx, gain = result['L0'], result['Lcross'], result['A']
    if l0 is None or lx is None:
        return None, 'unreachable_normal_or_crossing'
    if max(l0, lx) > 48:
        return None, 'path_above48'
    length = next((name for name, lo, hi in (('short',10,17),('medium',18,25),('long',26,33)) if lo <= l0 <= hi), None)
    if length is None:
        return None, 'length_outside_bins'
    value = 'large_gain' if gain >= 4 else 'small_gain' if gain >= 1 else 'harmful' if gain <= -2 else None
    if value is None:
        return None, 'A_zero_or_minus_one'
    return (length, value), None


def kernel_state(f):
    import jax.numpy as jnp
    from hackrl import spatial_wall_pass as env
    return env.make_spatial_wall_pass_state(0).replace(
        walkable=jnp.asarray(f['walkable'],bool),wall_mask=jnp.asarray(f['wall_mask'],bool),
        camp_position=jnp.asarray(f['camp'],jnp.int32),player_position=jnp.asarray(f['start'],jnp.int32),
        item_position=jnp.asarray(f['item'],jnp.int32),player_direction=jnp.int32(f['direction']))


def from_state(state, family='historical'):
    return dict(walkable=np.asarray(state.walkable).tolist(),wall_mask=np.asarray(state.wall_mask).tolist(),
                camp=np.asarray(state.camp_position).tolist(),start=np.asarray(state.player_position).tolist(),
                item=np.asarray(state.item_position).tolist(),direction=int(state.player_direction),family=family)


def verify(f, answer):
    """Orthogonal check against the real kernel, not a policy or learned scorer."""
    import jax
    import jax.numpy as jnp
    from hackrl import spatial_wall_pass as env
    initial = kernel_state(f)
    graphs = [route_graph(f, False), route_graph(f, True)]
    fixed_paths, mutant_paths = [g.paths() for g in graphs]
    assert fixed_paths[1] is None
    assert fixed_paths[0] is not None and len(fixed_paths[0]) == answer['L0']
    assert [None if x is None else len(x) for x in mutant_paths] == [answer['L0'], answer['Lcross']]
    # Outside a bug-triggering edge, all graph transitions agree exactly.
    # Removing crossings retains the same shortest successful normal distance;
    # failed fixed dashes can additionally turn in place, without shortening it.
    fixed, mutant = graphs
    same = ~mutant.transition_fields['wall_pass']
    assert np.array_equal(fixed.targets[same], mutant.targets[same])
    checks = []
    kernel_outputs = []
    for g, kernel in zip(graphs, ('fixed','mutant')):
        n = len(g.nodes)
        p, direction, carrying, crossed = g.nodes.T
        states = jax.tree.map(lambda x: jnp.broadcast_to(x,(n*23,*x.shape)),initial)
        states = states.replace(player_position=jnp.asarray(np.repeat(g.positions[p],23,axis=0)),
            player_direction=jnp.asarray(np.repeat(direction+1,23)),
            carrying_item=jnp.asarray(np.repeat(carrying.astype(bool),23)),
            item_present=jnp.asarray(np.repeat(~carrying.astype(bool),23)),
            episode_wall_passed=jnp.asarray(np.repeat(crossed.astype(bool),23)),
            item_collected_after_wall_pass=jnp.asarray(np.repeat((carrying & crossed).astype(bool),23)))
        actions = jnp.tile(jnp.arange(23),n)
        out,tr = _kernel_batch(kernel)(states,actions)
        kernel_outputs.append(jax.device_get((out,tr)))
        actual = dict(position=out.player_position,direction=out.player_direction,carrying=out.carrying_item,
                      crossed=out.episode_wall_passed,delivered=tr.delivered,picked_up=tr.picked_up,
                      wall_pass=tr.crossed_wall,dash_succeeded=tr.dash_succeeded)
        for name, value in actual.items():
            expected = g.transition_fields[name].reshape(np.asarray(value).shape)
            np.testing.assert_array_equal(np.asarray(value),expected,err_msg=name)
        checks.append(dict(kernel=kernel,transitions=n*23))
    # Every full kernel state and transition field agrees off the fault edges,
    # including tick and provenance fields omitted from the search key.
    outside_fault = same.ravel()
    for left,right in zip(jax.tree.leaves(kernel_outputs[0]),jax.tree.leaves(kernel_outputs[1])):
        np.testing.assert_array_equal(left[outside_fault],right[outside_fault])
    # Prohibiting a mutant crossing means its physical displacement is rejected;
    # the requested direction still changes, exactly as in fixed.
    rejected = mutant.targets.copy()
    n = len(fixed.nodes)
    rows,actions = np.nonzero(~same)
    old_p,_,carry,cross = fixed.nodes[rows].T
    rejected[rows,actions] = (((old_p*4+(actions-17))*2+carry)*2+cross)
    np.testing.assert_array_equal(rejected,fixed.targets)
    replay = []
    for kernel, actions in (('fixed',answer['normal_actions']),('mutant',answer['crossing_actions'])):
        for delay in (0,12):
            state = initial
            rewards = []
            crossed = False
            for action in [0]*delay + actions:
                before = state
                state,tr = _kernel_one(kernel)(state,jnp.int32(action))
                rewards.append(int(tr.delivered))
                crossed |= bool(tr.crossed_wall)
            assert int(state.delivered_total)==1 and int(state.tick)==delay+len(actions)<=64
            assert sum(rewards)==1 and rewards[-1]==1
            assert crossed == (kernel=='mutant')
            observation,reward_state,reward,_,_ = _wrapper_step(kernel)(before,jnp.int32(actions[-1]))
            assert float(reward)==1.0
            for left,right in zip(jax.tree.leaves(state),jax.tree.leaves(reward_state)):
                np.testing.assert_array_equal(left,right)
            replay.append(dict(kernel=kernel,delay=delay,ticks=int(state.tick),success=True,wall_pass=crossed,
                               control_return=env.DISCOUNT**(len(actions)-1),
                               episode_return=env.DISCOUNT**(len(actions)+delay-1)))
    # Variant is a step parameter, not a state/observation field.
    obs = env.encode_spatial_wall_pass_observation(env.observe_spatial_wall_pass(initial))
    assert np.isfinite(np.asarray(obs)).all()
    fixed_noop = _wrapper_step('fixed')(initial,jnp.int32(0))[0]
    mutant_noop = _wrapper_step('mutant')(initial,jnp.int32(0))[0]
    for left,right in zip(jax.tree.leaves(fixed_noop),jax.tree.leaves(mutant_noop)):
        np.testing.assert_array_equal(left,right)
    return dict(transition_checks=checks,replay=replay,initial_observation_sha256=hashlib.sha256(np.asarray(obs).tobytes()).hexdigest())


from functools import lru_cache


@lru_cache(None)
def _kernel_batch(kernel):
    import jax
    from hackrl import spatial_wall_pass as env
    return jax.jit(jax.vmap(lambda s,a:env.spatial_wall_pass_step_with_transition(s,a,kernel)))


@lru_cache(None)
def _kernel_one(kernel):
    import jax
    from hackrl import spatial_wall_pass as env
    return jax.jit(lambda s,a:env.spatial_wall_pass_step_with_transition(s,a,kernel))


@lru_cache(None)
def _wrapper_step(kernel):
    import jax
    from hackrl import spatial_wall_pass as env
    wrapper=env.SpatialWallPassEnvNoAutoReset(kernel)
    return jax.jit(lambda s,a:wrapper.step(jax.random.PRNGKey(0),s,a))
