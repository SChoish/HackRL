import json
from pathlib import Path

import numpy as np
import pytest

from hackrl import spatial_layout_panel as p
from hackrl import spatial_wall_pass as env


def reference():
    return p.from_state(env.make_spatial_wall_pass_state(0))


def test_known14_and8_and_all_kernel_actions():
    f = reference()
    result = p.solve(f)
    assert (result['L0'], result['Lcross'], result['A']) == (14,8,6)
    check = p.verify(f,result)
    assert len(check['transition_checks'])==2 and len(check['replay'])==4


def test_nonmovement_actions_cannot_create_shortcuts():
    g = p.route_graph(reference(),True)
    for action in (0,5,6,7,8,9,10,11,12,13,14,15,16):
        np.testing.assert_array_equal(g.targets[:,action],np.arange(len(g.nodes)))


def test_direction_is_part_of_state_and_updates_on_failed_dash():
    f = reference(); f['start']=[9,8]; f['direction']=1
    g=p.route_graph(f,False)
    target=g.targets[g.initial,18]
    assert target!=g.initial
    assert g.nodes[target,0]==g.nodes[g.initial,0]
    assert g.nodes[target,1]==1 # zero-based RIGHT


def test_harmful_crossing_is_not_negative_mutant_optimum_gain():
    f=p.fixture([(r,9) for r in range(7,12)],[9,6],[9,7],[7,7])
    r=p.solve(f)
    assert r['L0']<r['Lcross'] and r['A']<0
    paths=p.route_graph(f,True).paths()
    assert min(map(len,paths))==r['L0']


def test_classification_and48_limit():
    assert p.classify(dict(L0=26,Lcross=30,A=-4))[0]==('long','harmful')
    assert p.classify(dict(L0=10,Lcross=9,A=1))[0]==('short','small_gain')
    assert p.classify(dict(L0=10,Lcross=10,A=0))[1]=='A_zero_or_minus_one'
    assert p.classify(dict(L0=26,Lcross=49,A=-23))[1]=='path_above48'
    assert p.classify(dict(L0=None,Lcross=8,A=None))[1]=='unreachable_normal_or_crossing'


def test_canonical_translation_and_d4_including_direction():
    original=reference(); digest=p.canonical_hash(original)
    for split in ('train','validation','test'):
        for i in range(16):
            f=p.from_state(env.make_spatial_wall_pass_state(i,split=split))
            assert p.canonical_hash(f)==digest
    changed={**original,'direction':3}
    assert p.canonical_hash(changed)!=digest
    assert p.canonical_hash(changed,include_direction=False)==p.canonical_hash(original,include_direction=False)


def test_generator_deterministic_same_support_and_unique_roles():
    r1=np.random.default_rng(20261012); r2=np.random.default_rng(20261012)
    for i in range(16):
        first=p.generate_candidate(r1,i); second=p.generate_candidate(r2,i)
        assert first==second and first['family']==p.FAMILIES[i%4]
        assert set(map(tuple,np.argwhere(np.asarray(first['walkable'])|first['wall_mask'])))==p.SUPPORT
        assert len({tuple(first[k]) for k in ('camp','start','item')})==3


def test_historical_transfer_fixed_and_noncross_graph_consistency():
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'docs/manifests/spatial_wall_pass_frozen_transfer_v1.json').read_text())
    for old in spec['fixtures']:
        f=p.fixture(old['walls'],old['camp'],old['start'],old['item'])
        r=p.solve(f)
        fixed,mutant=p.route_graph(f,False).paths(),p.route_graph(f,True).paths()
        assert len(fixed[0])==r['L0'] and fixed[1] is None
        assert min(map(len,mutant))==min(r['L0'],r['Lcross'])


def test_layout_gate_is_not_evaluation_authorization():
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'docs/manifests/spatial_layout_panel_v1.json').read_text())
    assert spec['authorization']==dict(panel_generation=True,policy_evaluation=False,training=False)
    assert spec['candidate_cap']==50000 and spec['generator_seed']==20261012
