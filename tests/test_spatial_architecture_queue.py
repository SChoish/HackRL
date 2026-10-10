"""Execution contract tests; no production training is launched here."""
from pathlib import Path
import json
import sys
import jax
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_spatial_architecture as run
from hackrl import spatial_architecture as a
from hackrl import spatial_time_history_eval as evaluation
from hackrl import spatial_wall_pass_gc as gc


def test_jobs_budgets_and_retention():
    jobs=list(run.jobs())
    assert len(jobs)==len(set(jobs))==60
    assert jobs[:3]==[('current_mlp',132,None),('current_mlp',132,0.),('current_mlp',132,.1)]
    assert jobs[12]==('current_mlp',133,None)
    total=sum(run.config(s,q).num_updates*run.config(s,q).batch_size for _,s,q in jobs)
    assert total==run.SPEC['planned_collector_slots']
    retained=sum(len(run.SPEC['evaluation_updates']['pretrain' if q is None else 'adaptation']) for _,_,q in jobs)
    assert retained==run.SPEC['production_storage']['planned_retained_checkpoints']==420


def test_checkpoint_integrity_and_partial_directory_filter(tmp_path,monkeypatch):
    import shutil
    assert shutil.disk_usage(tmp_path).free>8*1024**3
    assert shutil.disk_usage('/raid/ext_csv').free>8*1024**3
    monkeypatch.setattr(run,'ROOT',tmp_path)
    monkeypatch.setattr(run,'capacity',lambda event:None)
    monkeypatch.setattr(run,'provenance',lambda:dict(execution_sha='test-only'))
    c=gc.SpatialWallPassGCConfig(num_envs=16,num_steps=4,minibatch_size=16,policy_hidden_size=16,teacher_hidden_size=16,num_updates=512)
    _,r,_,l,_=a.initialize(c,'current_mlp',small_test=True)
    path=tmp_path/'cell/checkpoints/update_0'
    run.save(path,'current_mlp',c,None,r,l,None,dict(valid_learner_samples=0,collector_slots=0))
    assert run.complete_checkpoints(path.parent,'current_mlp',c,None)=={0:path}
    (path.parent/'update_32.pending.example').mkdir()
    assert run.complete_checkpoints(path.parent,'current_mlp',c,None)=={0:path}
    with pytest.raises(RuntimeError,match='overwrite'):
        run.save(path,'current_mlp',c,None,r,l,None,{})
    with pytest.raises(AssertionError):run.validate_checkpoint(path,'gru',c,None)
    # A metadata hash mismatch must close resume, not silently skip this checkpoint.
    original=json.loads((path/'metadata.json').read_text())
    run.write(path/'metadata.json',dict(original,state_sha256='invalid'))
    with pytest.raises(AssertionError):run.complete_checkpoints(path.parent,'current_mlp',c,None)


def test_cached_evaluator_matches_validated_path():
    from test_spatial_time_history import NoopNetwork
    model=NoopNetwork()
    expected=evaluation.evaluate(model,{},132,history_api=a)
    cached=run.Evaluator(model)
    actual=cached({},132)
    assert actual==expected
    assert cached({},132)==actual # fresh state/RNG even with compiled code reused
    assert sum(len(v['episodes']) for mode in actual.values() for v in mode.values())==4752


def test_learning_code_still_matches_gpu_validation():
    path=Path(run.SPEC['validation_root'])/'result.json'
    if not path.exists():pytest.skip('RAID validation artifacts not available')
    result=json.loads(path.read_text())
    assert run.sha(path)==run.SPEC['validation_evidence']['result_sha256']
    for name,digest in result['provenance']['source_file_sha256'].items():
        if name.startswith('src/'):assert run.sha(run.REPO/name)==digest


def test_full_queue_lifecycle_and_resume(tmp_path,monkeypatch):
    from dataclasses import replace
    import shutil
    assert shutil.disk_usage(tmp_path).free>8*1024**3
    assert shutil.disk_usage('/raid/ext_csv').free>8*1024**3
    monkeypatch.setattr(run,'ROOT',tmp_path)
    monkeypatch.setattr(run,'capacity',lambda event:None)
    monkeypatch.setattr(run,'provenance',lambda:dict(execution_sha='test-only'))
    monkeypatch.setattr(run,'SPEC',{**run.SPEC,'pretrain_updates':1,'adapt_updates':1,
                                  'evaluation_updates':{'pretrain':[0,1],'adaptation':[0,1]}})
    c=gc.SpatialWallPassGCConfig(num_envs=2,num_steps=64,minibatch_size=64,policy_hidden_size=16,teacher_hidden_size=16,num_updates=1,seed=132)
    monkeypatch.setattr(run,'config',lambda seed,q:c if q is None else replace(c,goal_mode='delivery',variant='mutant'))
    initialize=a.initialize
    monkeypatch.setattr(a,'initialize',lambda c,arch:initialize(c,arch,small_test=True))
    # Only skip evaluation computation here; cached-vs-original equality is separately tested.
    result={s:{'fixed_delay0':{'summary':{'success_rate':0.0}}} for s in ('mode','sample')}
    monkeypatch.setattr(run.Evaluator,'__call__',lambda self,params,seed:result)
    for q in (None,0.,.1):
        assert run.run_cell('current_mlp',132,q)
        cell=run.cell_path('current_mlp',132,q)
        before=json.loads((cell/'summary.json').read_text())
        journal=(cell/'attempts.jsonl').read_text()
        assert run.run_cell('current_mlp',132,q)
        assert json.loads((cell/'summary.json').read_text())==before
        assert (cell/'attempts.jsonl').read_text()==journal # completed work not rerun
        assert before['collection_accounting']['collector_slots']==128
    first=json.loads((run.cell_path('current_mlp',132,0.)/'origin.json').read_text())
    second=json.loads((run.cell_path('current_mlp',132,.1)/'origin.json').read_text())
    assert first['parent_pretrain_sha256']==second['parent_pretrain_sha256']
