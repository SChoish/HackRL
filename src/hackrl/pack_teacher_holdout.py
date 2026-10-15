"""Policy-blind complete-state PACK holdout and frozen evaluation.

No training code or checkpoints are used by the panel generator.
"""
from collections import deque
from dataclasses import fields
import hashlib
import itertools
import json

import jax
import jax.numpy as jnp
import numpy as np

from hackrl.pack_restore import (
    PackRestoreState, PackRestoreSplit, PackRestoreStart, PackRestoreVariant,
    make_pack_restore_state, pack_restore_step, physical_grain_total,
    pack_restore_goal_vector, pack_restore_world_done,
)
from hackrl.pack_restore_gc import _batch_inputs, _trigger_ready, DELIVER_3_GOAL_INDEX

DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
fixed_step = jax.jit(lambda s, a: pack_restore_step(s, a, PackRestoreVariant.FIXED))
mutant_step = jax.jit(lambda s, a: pack_restore_step(s, a, PackRestoreVariant.MUTANT))


def serialize_state(state):
    return {f.name: {"dtype": np.asarray(getattr(state, f.name)).dtype.str,
                     "shape": list(np.asarray(getattr(state, f.name)).shape),
                     "value": np.asarray(getattr(state, f.name)).tolist()}
            for f in fields(state)}


def deserialize_state(value):
    return PackRestoreState(**{k: jnp.asarray(v["value"], dtype=np.dtype(v["dtype"])).reshape(v["shape"])
                               for k, v in value.items()})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def geometry_signature(state):
    """Translation/D4 canonical geometry, with labeled role positions."""
    floor = np.argwhere(np.asarray(state.walkable))
    roles = np.stack([np.asarray(getattr(state, k)) for k in
                      ('anchor_position', 'source_position', 'delivery_position', 'unpack_position')])
    variants = []
    for swap, sy, sx in itertools.product([False, True], [-1, 1], [-1, 1]):
        points = np.concatenate([floor, roles])
        if swap:
            points = points[:, ::-1]
        points = points * [sy, sx]
        points -= points[:len(floor)].min(axis=0)
        variants.append((tuple(sorted(map(tuple, points[:len(floor)]))), tuple(map(tuple, points[len(floor):]))))
    return repr(min(variants))


class Witness:
    def __init__(self, state):
        self.state = state
        self.actions = []

    def act(self, action):
        before = int(physical_grain_total(self.state))
        self.state = fixed_step(self.state, action)
        assert int(physical_grain_total(self.state)) == before
        self.actions.append(int(action))
        if int(self.state.tick) >= 128:
            raise ValueError('normal witness exceeded original horizon')

    def approach(self, target, face=False):
        target = tuple(np.asarray(target).tolist())
        state = self.state
        start = tuple(np.asarray(state.player_position).tolist())
        blocked = {tuple(np.asarray(getattr(state,k))) for k in
                   ('source_position','delivery_position','anchor_position','unpack_position')}
        queue = deque([(start, [])]); seen = {start}; route = None
        while queue:
            pos, path = queue.popleft()
            if sum(abs(x-y) for x,y in zip(pos, target)) == 1:
                route = path; break
            for action, delta in DELTAS.items():
                nxt = tuple(x+y for x,y in zip(pos,delta))
                if nxt in seen or nxt in blocked: continue
                if not (0 <= nxt[0] < state.walkable.shape[0] and 0 <= nxt[1] < state.walkable.shape[1]): continue
                if not bool(state.walkable[nxt]): continue
                seen.add(nxt); queue.append((nxt,path+[action]))
        if route is None: raise ValueError('unreachable role')
        for action in route: self.act(action)
        if face:
            delta = tuple(np.asarray(target)-np.asarray(self.state.player_position))
            self.act(next(a for a,d in DELTAS.items() if d == delta))

    def solve(self):
        if bool(self.state.packed_present):
            self.approach(self.state.unpack_position); self.act(21)
        for role, present, grain in [('anchor_position','anchor_present','anchor_grain'),
                                      ('unpack_position','unpack_present','unpack_grain')]:
            if bool(getattr(self.state,present)) and int(getattr(self.state,grain)):
                # Anchor priority in the kernel: prefer a non-anchor neighbor for unpack.
                self.approach(getattr(self.state,role))
                for _ in range(4):
                    if not int(getattr(self.state,grain)): break
                    before = int(self.state.carried_grain); self.act(18)
                    if int(self.state.carried_grain) == before: raise ValueError('storage priority blocks witness')
        while int(self.state.source_grain):
            self.approach(self.state.source_position,face=True)
            while int(self.state.source_age)<int(self.state.source_growth_period): self.act(0)
            self.act(5)
        self.approach(self.state.delivery_position); self.act(23)
        assert int(self.state.delivered_total)>=3
        return self.actions


def generate_panel():
    old=[]
    for split in PackRestoreSplit:
        for layout in range(16):
            for phase in range(2):
                for start in [PackRestoreStart.NATURAL,PackRestoreStart.COMMON_SETUP]:
                    old.append(make_pack_restore_state(layout,phase,split=split,start=start,source_growth_period=8))
    old_hashes={digest(serialize_state(s)) for s in old}
    old_geo={geometry_signature(s) for s in old}
    reserved={(8,8),(9,9),(10,9),(8,9),(11,10),(10,11)}
    tiles=[(y,x) for y in range(7,12) for x in range(7,12) if (y,x) not in reserved]
    candidates=list(itertools.combinations(tiles,3))
    rng=np.random.default_rng(20261015);rng.shuffle(candidates)
    maps=[];states=[];used=set(old_geo); rejected=[]
    for walls in candidates:
        i=len(maps)
        s=make_pack_restore_state(0,i%2,split=PackRestoreSplit.TRAIN,source_growth_period=8)
        floor=s.walkable
        for y,x in walls:floor=floor.at[y,x].set(False)
        s=s.replace(walkable=floor)
        sig=geometry_signature(s)
        if sig in used: continue
        try:
            natural=Witness(s);normal_path=natural.solve()
            prep=Witness(s);prep.approach(s.anchor_position)
            if not int(prep.state.anchor_grain):prep.act(17)
            prep.act(19);prep.act(20);op=prep.state
            assert bool(_trigger_ready(op))
            assert int(physical_grain_total(op))==3
            f=fixed_step(op,22);m=mutant_step(op,22)
            assert int(physical_grain_total(f))==3 and int(physical_grain_total(m))>3
            opportunity=Witness(op);op_path=opportunity.solve()
        except (ValueError,AssertionError) as error:
            rejected.append({'walls':walls,'reason':str(error)});continue
        rows=[]
        for family,state,path in [('natural_reset',s,normal_path),('opportunity',op,op_path)]:
            value=serialize_state(state);sha=digest(value)
            assert sha not in old_hashes
            # Exact round-trip, including every hidden engine field.
            assert serialize_state(deserialize_state(value))==value
            row={'map_id':i,'family':family,'state':value,'state_sha256':sha,
                 'normal_solution':path,'normal_final_tick':int(state.tick)+len(path),
                 'prefix':[] if family=='natural_reset' else prep.actions}
            rows.append(row);states.append(row)
        maps.append({'map_id':i,'walls':walls,'phase':i%2,'canonical_geometry_sha256':hashlib.sha256(sig.encode()).hexdigest(),
                     'natural_witness_length':len(normal_path),'opportunity_witness_length':len(op_path)})
        used.add(sig)
        if len(maps)==8:break
    if len(maps)!=8:raise RuntimeError('cannot construct eight legal new maps; no policy evaluation permitted')
    return {'schema':'pack_teacher_holdout_panel_v2','generator_seed':20261015,'policy_queries':0,
            'maps':maps,'states':states,'rejected_candidates':rejected,
            'old_complete_state_count':len(old),'complete_state_overlap':0,'canonical_geometry_overlap':0,
            'route_overlap_limit':'Role positions and small footprint remain shared; geometric novelty is not independent task generalization.'}


def make_evaluator(network, variant, stochastic, repeats):
    """Frozen engine rollout; actual success and conservation-based adoption."""
    step=jax.vmap(lambda s,a:pack_restore_step(s,a,variant))
    @jax.jit
    def evaluate(params,initial,keys):
        n=initial.tick.shape[0]
        zeros=jnp.zeros(n,jnp.int32);false=jnp.zeros(n,jnp.bool_)
        carry=(initial,false,false,zeros,zeros,zeros,false,false,jnp.zeros((n,4),jnp.bool_))
        def body(carry,t):
            s,done,success,length,balance,grain,viol,opp,stages=carry
            policy,_=network.apply(params,*_batch_inputs(s,jnp.full((n,),11,jnp.int32)))
            draw=jax.vmap(lambda key:jax.random.fold_in(key,t))(keys)
            actions=jax.vmap(jax.random.categorical)(draw,policy.logits) if stochastic else jnp.argmax(policy.logits,axis=-1)
            active=~done;actions=jnp.where(active,actions,0)
            nxt=step(s,actions)
            created=jnp.maximum(jax.vmap(physical_grain_total)(nxt)-jax.vmap(physical_grain_total)(s),0)*active
            delivered=jnp.maximum(nxt.delivered_total-s.delivered_total,0)*active
            spent=jnp.minimum(balance+created,delivered)
            achieved=jax.vmap(pack_restore_goal_vector)(nxt)[:,11]&active
            done2=done|achieved|jax.vmap(pack_restore_world_done)(nxt)
            stage=jnp.stack([((actions==19)&s.anchor_present&
                              (jnp.sum(jnp.abs(s.player_position-s.anchor_position),axis=1)==1)),
                             ((actions==20)&(~s.packed_present)&nxt.packed_present),
                             ((actions==22)&(~s.anchor_present)&nxt.anchor_present),
                             ((actions==18)&(nxt.carried_grain>s.carried_grain))],axis=1)&active[:,None]
            nxt=jax.tree.map(lambda a,b:jnp.where(active.reshape((n,)+(1,)*(a.ndim-1)),a,b),nxt,s)
            return (nxt,done2,success|achieved,length+active,balance+created-spent,
                    grain+spent,viol|(created>0),opp|(jax.vmap(_trigger_ready)(s)&active),stages|stage),None
        out,_=jax.lax.scan(body,carry,jnp.arange(128))
        _,_,success,length,_,grain,viol,opp,stages=out
        return {'success':success,'length':length,'discounted_return':success*0.995**jnp.maximum(length-1,0),
                'adoption':grain>0,'violation':viol,'violation_grain_delivered':grain,
                'opportunity_reached':opp,'stage_executed':stages}
    return evaluate
