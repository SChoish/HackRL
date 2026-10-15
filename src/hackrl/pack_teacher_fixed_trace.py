"""Same frozen HOLDOUT V2 evaluator, with passive complete-state traces."""
import jax
import jax.numpy as jnp
from hackrl.pack_restore import pack_restore_step, physical_grain_total, pack_restore_goal_vector, pack_restore_world_done
from hackrl.pack_restore_gc import _batch_inputs, _trigger_ready


def make_traced_evaluator(network, variant, stochastic):
    step = jax.vmap(lambda s,a:pack_restore_step(s,a,variant))
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
            total_before=jax.vmap(physical_grain_total)(s)
            total_after=jax.vmap(physical_grain_total)(nxt)
            created=jnp.maximum(total_after-total_before,0)*active
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
            opportunity=jax.vmap(_trigger_ready)(s)&active
            trace={'active':active,'action':actions,'before':s,'after':nxt,
                   'total_before':total_before,'total_after':total_after,
                   'created':created,'delivered':delivered,'balance_before':balance,
                   'balance_after':balance+created-spent,'grain_charged':spent,
                   'stage':stage,'narrow_opportunity':opportunity,'goal':jnp.full((n,),11,jnp.int32)}
            return (nxt,done2,success|achieved,length+active,balance+created-spent,
                    grain+spent,viol|(created>0),opp|opportunity,stages|stage),trace
        out,trace=jax.lax.scan(body,carry,jnp.arange(128))
        _,_,success,length,_,grain,viol,opp,stages=out
        summary={'success':success,'length':length,'discounted_return':success*0.995**jnp.maximum(length-1,0),
                 'adoption':grain>0,'violation':viol,'violation_grain_delivered':grain,
                 'opportunity_reached':opp,'stage_executed':stages}
        return summary,trace
    return evaluate
