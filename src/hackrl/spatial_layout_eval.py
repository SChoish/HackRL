"""Frozen history-aware evaluation with externally locked initial states."""
import distrax
import jax
import jax.numpy as jnp
import numpy as np
from hackrl import spatial_architecture as a
from hackrl import spatial_wall_pass as env
from hackrl import spatial_wall_pass_gc as gc
from hackrl.spatial_recovery_probe import intervention_step, records_from_trace


def make_rollout(network, stochastic=False, repeats=1):
    @jax.jit
    def rollout(params, learner_seed, fixed, delay, initial16, map_ids16):
        initial=jax.tree.map(lambda x:jnp.repeat(x,repeats,axis=0),initial16)
        ids=jnp.repeat(map_ids16,repeats)
        repeat_ids=jnp.tile(jnp.arange(repeats,dtype=jnp.int32),map_ids16.shape[0])
        n=map_ids16.shape[0]*repeats
        goals=jnp.full(n,gc.DELIVERY_GOAL_INDEX,jnp.int32)
        seeds=(61000 if stochastic else 60000)+1000*learner_seed
        seeds=seeds+(64 if stochastic else 1)*ids+repeat_ids
        keys=jax.vmap(jax.random.PRNGKey)(seeds)

        def step(carry,t):
            state,history,done,keys=carry
            pi,_=network.apply(params,*a.pack_inputs(state,goals,history))
            splits=jax.vmap(lambda k:jax.random.split(k,2))(keys)
            if stochastic:
                choice=jax.vmap(lambda logits,key:distrax.Categorical(logits=logits).sample(seed=key))(pi.logits,splits[:,1])
            else:
                choice=jnp.argmax(pi.logits,axis=-1)
            active=~done;forced=jnp.full(n,t<delay)&active;controlled=active&~forced
            action=jnp.where(controlled,choice,0)
            out,tr,_,triggered,blocked=jax.vmap(lambda s,ac:intervention_step(s,ac,fixed,False,False))(state,action)
            delivered=active&tr.delivered
            trace=dict(active=active,forced=forced,action=action,position=state.player_position,
                next_position=out.player_position,wall_pass=active&tr.crossed_wall,blocked=active&blocked,
                trigger=active&triggered,delivery=delivered,beneficial=delivered&out.delivered_after_wall_pass,
                dash=active&tr.dash_requested,clear_dash=active&tr.dash_succeeded&~tr.crossed_wall)
            history=a.advance_history(history,state,action,controlled,active&(out.tick>=64))
            return (gc._tree_where(active,out,state),history,done|delivered|(out.tick>=64),
                    jnp.where(controlled[:,None],splits[:,0],keys)),trace

        _,trace=jax.lax.scan(step,(initial,a.empty_history(initial),jnp.zeros(n,bool),keys),jnp.arange(64,dtype=jnp.int32))
        return trace
    return rollout


def decorate(trace, maps, repeats, delay, kernel, selection):
    rows=records_from_trace(trace,repeats,delay)
    for r in rows:
        m=maps[r['layout_index']]
        l0,lx=m['oracle']['L0'],m['oracle']['Lcross']
        optimum=l0 if kernel=='fixed' else min(l0,lx)
        oracle_return=env.DISCOUNT**(optimum-1) if optimum+delay<=64 else 0.0
        control=env.DISCOUNT**(r['control_length']-1) if r['success'] else 0.0
        assert control<=oracle_return+1e-12, 'Policy exceeded verified shortest path'
        assert abs(r['discounted_return']-env.DISCOUNT**delay*control)<1e-12
        r.update(map_id=m['map_id'],cell='/'.join(m['cell']),family=m['fixture']['family'],
                 delay=delay,kernel=kernel,selection=selection,control_return=control,
                 oracle_control_return=oracle_return,regret=oracle_return-control,
                 oracle_length=optimum,normal_length=l0,crossing_length=lx)
        if selection=='sample':r.pop('path')
    return rows


def pair_rows(records):
    pairs={}
    for r in records:
        key=(r['selection'],r['delay'],r['map_id'],r['repeat'])
        assert r['kernel'] not in pairs.setdefault(key,{})
        pairs[key][r['kernel']]=r
    out=[]
    for (selection,delay,map_id,repeat),both in pairs.items():
        assert set(both)=={'fixed','mutant'}
        f,m=both['fixed'],both['mutant']
        out.append(dict(selection=selection,delay=delay,map_id=map_id,repeat=repeat,cell=m['cell'],family=m['family'],
            fixed_success=float(f['success']),mutant_success=float(m['success']),crossing=float(m['wall_pass']),
            joint=float(f['success'] and m['success'] and m['wall_pass']),
            fixed_control_return=f['control_return'],mutant_control_return=m['control_return'],
            control_return_gap=m['control_return']-f['control_return'],
            episode_return_gap=m['discounted_return']-f['discounted_return'],
            fixed_regret=f['regret'],mutant_regret=m['regret']))
    return out


METRICS=('fixed_success','mutant_success','crossing','joint','fixed_control_return','mutant_control_return',
         'control_return_gap','episode_return_gap','fixed_regret','mutant_regret')


def seed_aggregates(result):
    pairs=pair_rows(result['episodes'])
    out=[]
    groups={'primary_large':['short/large_gain'],'primary_small':['short/small_gain'],
            'primary_harmful':['short/harmful'],'primary_positive':['short/large_gain','short/small_gain'],
            'primary24':['short/large_gain','short/small_gain','short/harmful'],
            'extension8':['medium/large_gain']}
    for selection in sorted({p['selection'] for p in pairs}):
        for delay in (0,12):
            for group,cells in groups.items():
                items=[p for p in pairs if p['selection']==selection and p['delay']==delay and p['cell'] in cells]
                assert items
                # Every included cell has8maps, every map equal repeats.
                row=dict(architecture=result['architecture'],seed=result['seed'],fixed_percent=result['fixed_percent'],
                         selection=selection,delay=delay,group=group,maps=len({p['map_id'] for p in items}))
                row.update({k:float(np.mean([p[k] for p in items])) for k in METRICS})
                episodes=[e for e in result['episodes'] if e['selection']==selection and e['delay']==delay and e['cell'] in cells]
                for kernel in ('fixed','mutant'):
                    rs=[e for e in episodes if e['kernel']==kernel];ok=[e for e in rs if e['success']]
                    row[kernel+'_failure_episodes']=len(rs)-len(ok)
                    row[kernel+'_success_control_length']=float(np.mean([e['control_length'] for e in ok])) if ok else None
                out.append(row)
    return out


def primary_bootstrap(results, maps, repeats=10000, seed=20261013):
    """Paired seed-only and crossed seed/map bootstrap; no episode pseudo-n."""
    selected=[m for m in maps if m['cell'][0]=='short' and m['oracle']['A']>0]
    selected.sort(key=lambda m:(m['cell'][1],m['map_id']))
    assert len(selected)==16 and all(len([m for m in selected if m['cell'][1]==v])==8 for v in ('large_gain','small_gain'))
    matrices={}
    for arch in ('current_mlp','history_mlp'):
        rows=[]
        for learner in range(132,137):
            r=next(r for r in results if r['architecture']==arch and r['seed']==learner and r['fixed_percent']==10)
            ps={p['map_id']:p for p in pair_rows(r['episodes']) if p['selection']=='mode' and p['delay']==12}
            rows.append([[ps[m['map_id']][k] for k in METRICS] for m in selected])
        matrices[arch]=np.array(rows)
    difference=matrices['history_mlp']-matrices['current_mlp']
    rng=np.random.default_rng(seed)
    seed_ids=rng.integers(0,5,(repeats,5))
    seed_only=difference.mean(1)[seed_ids].mean(1)
    map_ids=np.concatenate((rng.integers(0,8,(repeats,8)),8+rng.integers(0,8,(repeats,8))),axis=1)
    two_way=difference[seed_ids[:,:,None],map_ids[:,None,:]].mean((1,2))
    return dict(comparison='history_mlp minus current_mlp; q10 mode delay12 primary_positive16',
        learner_seeds=list(range(132,137)),bootstrap_repeats=repeats,analysis_seed=seed,
        metrics={k:dict(paired_mean=float(difference[...,i].mean()),
            seed_differences=difference[...,i].mean(1).tolist(),
            seed_only95=np.percentile(seed_only[:,i],[2.5,97.5]).tolist(),
            two_way95=np.percentile(two_way[:,i],[2.5,97.5]).tolist()) for i,k in enumerate(METRICS)})
