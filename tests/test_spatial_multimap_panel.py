import copy
import numpy as np
from hackrl import spatial_layout_panel as p
from hackrl import spatial_multimap_panel as m

def test_group_ignores_endpoint_roles_and_direction():
    f=p.generate_candidate(np.random.default_rng(8),0);g=copy.deepcopy(f)
    g['start'],g['item']=g['item'],g['start'];g['direction']=4
    assert m.wall_group(f)==m.wall_group(g)
    assert m.split_for(m.wall_group(f))==m.split_for(m.wall_group(g))
    assert p.canonical_hash(f,include_direction=False)!=p.canonical_hash(g,include_direction=False)

def test_group_identifies_all_d4_and_translation():
    f=p.generate_candidate(np.random.default_rng(9),1)
    for flip in (False,True):
        for k in range(4):
            g=copy.deepcopy(f)
            for key in ('walkable','wall_mask'):
                a=np.asarray(f[key]);a=np.flip(a,axis=1) if flip else a
                g[key]=np.roll(np.rot90(a,k),-2,axis=0).tolist()
            assert m.wall_group(g)==m.wall_group(f)

def test_split_has_no_policy_or_endpoint_argument():
    assert {m.split_for(str(i)) for i in range(100)}=={'train','development','test'}
    assert all(m.split_for(str(i))==m.split_for(str(i)) for i in range(100))
