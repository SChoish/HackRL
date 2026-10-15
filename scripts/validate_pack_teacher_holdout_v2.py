#!/usr/bin/env python3
"""Two one-update GPU smoke branches from historical source; no holdout queries."""
import json
from pathlib import Path
import time
import jax
import numpy as np
import run_pack_teacher_holdout_v2 as run
import run_dual_leo_compare as base
from hackrl.dual_leo import load_dual_checkpoint


def main():
    assert jax.default_backend()=='gpu','GPU smoke requires actual GPU'
    run.verify();run.capacity()
    source=Path('/raid/ext_csv/HackRL/runs/pack_restore_scale_bc_v1/size_s/pack/dual/pretrain/seed30/checkpoints/update_512')
    original=run.sha(source/'state.msgpack');rows=[];initial=[]
    for condition,goals in [('D-on',None),('D-delivery',[11])]:
        job={'env':'pack','seed':30,'method':'dual','variant':'mutant','policy_hidden_size':512,'teacher_hidden_size':512,
             'pretrain_checkpoint':str(source),'origin_arm':{'learn_teacher':True,'imitate_teacher':True}}
        if goals is not None:job['teacher_goal_indices']=goals
        cfg=base._job_config(job,goal_mode='deliver_3',variant='mutant',updates=4096)
        network,template=base.ENVS['pack']['initialize'](cfg)
        teacher,leo_template,batch=base._start_teacher(base.ENVS['pack'],cfg,template)
        if condition=='D-delivery':
            prior=Path('/raid/ext_csv/HackRL/runs/pack_restore_delivery_only_teacher_v1/size_s/pack/dual_teacher_delivery_only/mutant/seed30/checkpoints/adapt_4096')
            old,old_leo=load_dual_checkpoint(prior,template,leo_template)
            assert int(old.global_update)==4608 and int(old_leo.step)>0
            assert np.all(np.asarray(old.current_goal)==11)
            assert jax.tree.structure(old.train_state.params)==jax.tree.structure(template.train_state.params)
            assert jax.tree.structure(old_leo.params)==jax.tree.structure(leo_template.params)
            assert all(a.shape==b.shape for a,b in zip(jax.tree.leaves(old.train_state.params),jax.tree.leaves(template.train_state.params)))
        runner,leo=base._load_adapt_start(run.ROOT,job,template,cfg)
        initial.append(run.tree_hash((runner,leo)))
        assert np.all(np.asarray(runner.current_goal)==11)
        update=base._dual_update(base.ENVS['pack'],network,teacher,cfg,batch,base._arm(job))
        t=time.monotonic();new,new_leo,metrics=update(runner,leo);jax.block_until_ready(new.train_state.step)
        host={k:float(np.asarray(v)) for k,v in metrics.items()}
        assert all(np.isfinite(v) for v in host.values())
        assert int(new.global_update)==513 and int(new.train_state.step)>int(runner.train_state.step)
        assert int(new_leo.step)>int(leo.step)
        assert host['teacher_target_head_count']==(12 if goals is None else 1)
        assert host['bc_policy_coef']>0 and host['learn_teacher']==1 and host['imitate_teacher']==1
        assert run.tree_hash(new.train_state.params)!=run.tree_hash(runner.train_state.params)
        assert run.tree_hash(new_leo.params)!=run.tree_hash(leo.params)
        # Real save/load test under the run's designated smoke directory only.
        dest=run.ROOT/'validation_checkpoints'/condition
        base._save_state(base.ENVS['pack'],'dual',dest,new,new_leo,cfg,base._arm(job))
        restored,restored_leo=load_dual_checkpoint(dest,template,leo_template)
        assert run.tree_hash((new,new_leo))==run.tree_hash((restored,restored_leo))
        assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves((new.train_state,new_leo)))
        rows.append({'condition':condition,'seconds_including_compile':time.monotonic()-t,
                     'ppo_steps':int(new.train_state.step)-int(runner.train_state.step),
                     'teacher_steps':int(new_leo.step)-int(leo.step),'metrics':host,'restore_identical':True})
        print(condition,rows[-1]['seconds_including_compile'],flush=True)
    assert initial[0]==initial[1] and run.sha(source/'state.msgpack')==original
    run.write(run.ROOT/'validation.json',{'passed':True,'backend':jax.default_backend(),'devices':[str(d) for d in jax.devices()],
                'source_unchanged':True,'paired_initial_state_sha256':initial[0],
                'execution_source_sha256':run.verify()['source_sha256'],
                'training_budget_charge':0,'separate_smoke_collector_slots':65536,'smoke_rows':rows})


if __name__=='__main__':main()
