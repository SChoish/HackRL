"""Policy-blind wall-group split contract for the existing small-map generator."""
import hashlib
import json
import numpy as np
from hackrl import spatial_layout_panel as panel


def wall_group(fixture):
    floor=np.argwhere(fixture['walkable']); walls=np.argwhere(fixture['wall_mask'])
    versions=[]
    for reflect in (False,True):
        for rotate in range(4):
            matrix=np.array(((1,0),(0,-1 if reflect else 1)))
            for _ in range(rotate):matrix=np.array(((0,1),(-1,0)))@matrix
            fl,wl=floor@matrix.T,walls@matrix.T
            minimum=np.concatenate((fl,wl)).min(0)
            versions.append(json.dumps([sorted(map(tuple,fl-minimum)),sorted(map(tuple,wl-minimum))],
                separators=(',',':'),default=int))
    return hashlib.sha256(min(versions).encode()).hexdigest()


def split_for(group):
    bucket=int(hashlib.sha256(('multimap-v1:'+group).encode()).hexdigest(),16)%10
    return 'train' if bucket<4 else 'development' if bucket<6 else 'test'


def historical(repo, prior_panel):
    # Import the existing policy-free exclusions; do not load any learner.
    from hackrl import spatial_wall_pass as env
    fixtures=[panel.from_state(env.make_spatial_wall_pass_state(i,split=s))
              for s in ('train','validation','test') for i in range(16)]
    old=json.loads((repo/'docs/manifests/spatial_wall_pass_frozen_transfer_v1.json').read_text())
    fixtures += [panel.fixture(x['walls'],x['camp'],x['start'],x['item']) for x in old['fixtures']]
    fixtures += [m['fixture'] for m in json.loads(prior_panel.read_text())['maps']]
    return fixtures


def validate_partition(maps, quotas, excluded):
    assert len(maps)==sum(3*n for n in quotas.values())
    groups={s:set() for s in quotas}; seen=set()
    for m in maps:
        f=m['fixture'];g=wall_group(f)
        assert g==m['wall_group_sha256'] and split_for(g)==m['split'] and g not in excluded
        groups[m['split']].add(g)
        geometry=panel.canonical_hash(f,include_direction=False)
        assert geometry not in seen and geometry==m['geometry_sha256'];seen.add(geometry)
        assert panel.classify(m['oracle'])[0]==('short',m['value_cell'])
    for s,n in quotas.items():
        for cell in ('large_gain','small_gain','harmful'):
            assert sum(m['split']==s and m['value_cell']==cell for m in maps)==n
        for other in quotas:
            if s!=other:assert not groups[s]&groups[other]
    return {s:len(g) for s,g in groups.items()}
