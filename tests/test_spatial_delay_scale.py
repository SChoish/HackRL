"""The scale adapter must preserve the completed S learning contract."""
from dataclasses import replace
from pathlib import Path
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import run_spatial_delay_scale as scale
import finalize_spatial_delay_scale as report

def test_registered_budget_and_evaluation_arithmetic():
    s=scale.SPEC
    n=len(s['seeds'])*len(s['new_sizes'])
    assert n*(1+len(s['fixed_probabilities']))==s['new_jobs']==30
    assert n*(s['pretrain_updates']+2*s['adapt_updates'])*s['num_envs']*s['num_steps']==s['new_collector_slots']
    assert n*len(s['pretrain_checkpoints'])*2*16*(1+64)==s['evaluation']['new_pretrain_episodes']
    assert n*2*len(s['adapt_checkpoints'])*11*3*16*(1+8)==s['evaluation']['new_adaptation_episodes']
    assert s['smoke']['total_update_calls']*512*64==s['smoke']['diagnostic_collector_slots']
    assert n*len(s['pretrain_checkpoints'])+n*2*len(s['adapt_checkpoints'])==s['storage']['retained_checkpoints']

@pytest.mark.parametrize('size',['M','L'])
def test_only_policy_width_changes_existing_configs(size):
    # Read-only comparison against the frozen S source configuration.
    if not scale.source_s(132).exists():pytest.skip('RAID source checkpoint unavailable')
    old=scale.pre_config('S',132)
    new=scale.pre_config(size,132)
    assert replace(new,policy_hidden_size=512)==old
    old_branch=scale.adapt.parent.config(132)
    branch=scale.branch_config(size,132)
    assert replace(branch,policy_hidden_size=512)==old_branch
    assert new.teacher_hidden_size==branch.teacher_hidden_size==512
    assert new.policy_hidden_size==branch.policy_hidden_size==scale.SPEC['policy_widths'][size]

def test_context_restores_bindings_even_after_failure():
    a=scale.adapt
    before=(a.ROOT,a.provenance,a.capacity,a.parent.templates,a.parent.source)
    with pytest.raises(ValueError):
        with scale.adaptation_context('M'):
            assert a.ROOT==scale.ROOT/'M'/'adaptation'
            assert a.parent.source(132)==scale.source('M',132)
            a.ROLL['sentinel']=object()
            raise ValueError('diagnostic')
    assert (a.ROOT,a.provenance,a.capacity,a.parent.templates,a.parent.source)==before
    assert not a.ROLL

def rows():
    return [dict(size=size,q=q,selection=mode,seed=seed,delay=d,
                 **{k:(seed-132)/4+d/100 for k in report.METRICS})
            for size in ('S','M','L') for q in (0.,.1) for mode in ('mode','sample')
            for seed in range(132,137) for d in (*range(9),10,12)]

def test_aggregation_keeps_all_seeds_and_equal_delay_weights():
    per_seed,groups=report.aggregate(rows())
    assert len(per_seed)==180 and len(groups)==36
    for g in groups:
        expected=.5+{'0':0,'1-8':.045,'10,12':.11}[g['group']]
        assert g['n_learners']==5
        assert g['mutant_success']==pytest.approx(expected)

@pytest.mark.parametrize('duplicate',[False,True])
def test_missing_or_duplicate_seed_delay_cannot_be_silently_dropped(duplicate):
    data=rows()
    if duplicate:data.append(data[0])
    else:data.pop(0)
    with pytest.raises(RuntimeError,match='Missing/duplicate'):
        report.aggregate(data)
