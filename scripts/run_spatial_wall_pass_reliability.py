#!/usr/bin/env python3
"""Locked 15-job episode-kernel reliability experiment; resumable RAID state."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_spatial_wall_pass_dual_adaptation as old
import jax
import numpy as np
from flax import serialization
from hackrl import spatial_wall_pass_gc as gc
from hackrl.dual_leo import load_dual_checkpoint, save_dual_checkpoint, make_dual_leo_update
from hackrl.spatial_wall_pass_reliability import attach_mixture, base_runner, make_mixed_outcome, mixture_metrics

REPO=Path(__file__).resolve().parents[1]
MANIFEST=REPO/"docs/manifests/spatial_wall_pass_reliability_mix_v1.json"
CONTRACT=json.loads(MANIFEST.read_text())
ROOT=Path(CONTRACT["run_root"])
SOURCE=Path(CONTRACT["source_root"])
SEEDS=tuple(CONTRACT["seeds"])
PROBS=tuple(CONTRACT["fixed_probabilities"])
UPDATES=tuple(CONTRACT["evaluation"]["updates"])
write=old._write_json

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(runner,leo):
    return hashlib.sha256(serialization.to_bytes({"gc":runner,"leo":leo})).hexdigest()

def config(seed,variant="mutant"):
    return replace(old._job_config({"seed":seed,"variant":variant}),sample_repeats_per_state=8)

def cell_name(seed,q):
    return f"fixed_{round(q*100):02d}pct_seed_{seed}"

def source(seed):
    return SOURCE/f"pretraining/dual_seed_{seed}/checkpoints/update_512"

def jobs():
    return [(seed,q) for seed in SEEDS for q in PROBS]

def check_contract():
    assert SEEDS==(132,133,134,135,136) and PROBS==(.01,.03,.1)
    assert CONTRACT["updates"]==4096 and CONTRACT["total_transitions"]==15*4096*512*64
    assert CONTRACT["evaluation"]["sample_repeats"]==8
    assert UPDATES==(0,32,128,256,512,1024,2048,4096)
    assert json.loads((SOURCE/"run_manifest.json").read_text())["execution_code_sha"]==CONTRACT["training_source_sha"]

def provenance():
    check_contract()
    if subprocess.check_output(["git","status","--porcelain"],cwd=REPO,text=True).strip():
        raise RuntimeError("Commit all execution sources before GPU smoke or production")
    paths=[str(p.relative_to(REPO)) for group in ("src/hackrl/*.py","scripts/run_spatial_wall_pass*.py",
        "scripts/run_spatial_wall_pass*.sh","docs/manifests/spatial_wall_pass_reliability_mix_v1.json")
        for p in REPO.glob(group)]
    paths += ["scripts/stop_identity_checked_gpu_keepalive.py",
              "scripts/guard_identity_checked_gpu_keepalive.py"]
    initial=jax.vmap(lambda i:gc.make_spatial_wall_pass_state(
        i,split=gc.SpatialWallPassSplit.VALIDATION))(jax.numpy.arange(16,dtype=jax.numpy.int32))
    rng_lists={str(seed):{selection:[base+1000*seed+stride*i+j
        for i in range(16) for j in range(repeats)]
        for selection,base,stride,repeats in (("mode",60000,1,1),("sample",61000,64,8))}
        for seed in SEEDS}
    return {"contract":CONTRACT,"execution_sha":subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        "manifest_sha256":sha(MANIFEST),"source_file_sha256":{p:sha(REPO/p) for p in paths},
        "evaluation_initial_state_sha256":hashlib.sha256(serialization.to_bytes(initial)).hexdigest(),
        "evaluation_rng_seeds":rng_lists,
        "evaluation_rng_sha256":hashlib.sha256(json.dumps(rng_lists,sort_keys=True).encode()).hexdigest(),
        "pretraining_state_sha256":{str(s):sha(source(s)/"state.msgpack") for s in SEEDS}}

def capacity(event):
    # Preserve all 120 science checkpoints, pending serialization overlap,
    # diagnostic/log allowance, and the user-mandated reserve.
    size=max((source(s)/"state.msgpack").stat().st_size for s in SEEDS)+1024*1024
    retained=120
    existing=sum(1 for p in ROOT.glob("fixed_*pct_seed_*/checkpoints/adapt_*/metadata.json")
                 if (p.parent/"state.msgpack").is_file())
    projected=max(0,retained-existing)*size+2*size+15*512*1024**2
    reserve=max(8*1024**3,int(np.ceil(.2*projected)))
    rec={"event":event,"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
         "chosen_filesystem":"/raid/ext_csv","representative_checkpoint_bytes":size,
         "retained_checkpoint_count":retained,"existing_checkpoint_count":existing,
         "projected_remaining_write_bytes":projected,"safety_reserve_bytes":reserve,
         "required_free_bytes":projected+reserve,
         "existing_run_bytes":sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()),
         "home":shutil.disk_usage("/home/ext_csv")._asdict(),"raid":shutil.disk_usage("/raid/ext_csv")._asdict()}
    if rec["raid"]["free"]<projected+reserve:
        raise RuntimeError("RAID peak plus reserve unavailable; stop at last completed checkpoint")
    with (ROOT/"capacity_checks.jsonl").open("a") as f:
        f.write(json.dumps(rec)+"\n")
    return rec

def templates(seed):
    c=config(seed)
    net,template,teacher,lt,mb=old._initialize_templates(c)
    # The attached plan intentionally changes evaluation repeats 64 -> 8.
    # Validate every inherited learning field using the original evaluation
    # count, then enforce this sole evaluation-budget difference explicitly.
    source_config=json.loads((source(seed)/"config.json").read_text())
    assert source_config["sample_repeats_per_state"]==64
    assert c.sample_repeats_per_state==8
    old._validate_branch_config(source(seed),replace(c,sample_repeats_per_state=64))
    r,l=load_dual_checkpoint(source(seed),template,lt)
    assert int(r.global_update)==512
    inherited=old._command_only_digest(r,l)
    r=gc.command_delivery(r)
    assert old._command_only_digest(r,l)==inherited
    # Both historic endpoints must originate in exactly this commanded full state.
    for branch in ("fixed_continue","mutant_adapt"):
        root=SOURCE/f"adaptation/{branch}_seed_{seed}"
        origin=json.loads((root/"origin.json").read_text())
        assert origin["source_state_sha256"]==sha(source(seed)/"state.msgpack")
        assert origin["adaptation_zero_state_sha256"]==digest(r,l)
        endpoint_config=json.loads((root/"checkpoints/adapt_0/config.json").read_text())
        current=gc.spatial_wall_pass_gc_config_payload(c)
        for key in current:
            if key not in ("variant","sample_repeats_per_state"):
                assert current[key]==endpoint_config[key],key
    return c,net,r,teacher,l,mb

def evaluate(net,r,c,update):
    before=serialization.to_bytes(r)
    result={"adaptation_update":update,"global_update":int(r.global_update)}
    for kernel in ("fixed","mutant"):
        result[kernel]={}
        for selection,repeats in (("mode",1),("sample",8)):
            result[kernel][selection]=gc.evaluate_spatial_wall_pass_gc_frozen(
                net,r.train_state.params,variant=kernel,stochastic=selection=="sample",
                repeats_per_state=repeats,seed_stride=64 if selection=="sample" else 1,
                seed_base=61000 if selection=="sample" else 60000,learner_seed=c.seed,
                record_episodes=True,diagnose_blocked_dash=kernel=="fixed",record_paths=selection=="mode")
    assert serialization.to_bytes(r)==before
    result["runner_state_immutable"]=True
    return result

def compare_trees(a,b,tolerance=0.):
    la,ta=jax.tree.flatten(a); lb,tb=jax.tree.flatten(b)
    assert ta==tb
    largest=0.
    for x,y in zip(la,lb):
        x=np.asarray(x);y=np.asarray(y)
        assert x.shape==y.shape
        if x.dtype.kind in "fc":
            assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
            delta=float(np.max(np.abs(x-y))) if x.size else 0.
            largest=max(largest,delta)
            assert delta<=tolerance,(delta,tolerance)
        else:
            np.testing.assert_array_equal(x,y)
    return largest

def smoke():
    c,net,r,teacher,l,mb=templates(132)
    records=[]
    for q,variant in ((0.,"mutant"),(1.,"fixed"),(.01,"mutant"),(.03,"mutant"),(.1,"mutant")):
        cc=replace(c,variant=variant)
        mixed=attach_mixture(r,132,q)
        update=jax.jit(make_dual_leo_update(net,teacher,cc,make_mixed_outcome(q),gc._batch_inputs,mb))
        out,leo,metrics=update(mixed,l)
        jax.block_until_ready(out.global_update)
        record={"q":q,"update":int(out.global_update),"mixture":mixture_metrics(out)}
        assert int(out.global_update)==513 and int(out.mixture.transitions.sum())==32768
        if q in (0.,1.):
            original=jax.jit(make_dual_leo_update(net,teacher,cc,gc.spatial_wall_pass_outcome,gc._batch_inputs,mb))
            baseline,bl,bm=original(r,l)
            record["endpoint_max_abs_difference"]=compare_trees(
                (base_runner(out),leo,metrics),(baseline,bl,bm),1e-6)
        # Round-trip the exact pre-update full mixture state, then replay the same update.
        restored=serialization.from_bytes(mixed,serialization.to_bytes(mixed))
        out2,leo2,metrics2=update(restored,l)
        record["resume_max_abs_difference"]=compare_trees((out,leo,metrics),(out2,leo2,metrics2))
        # No scientific checkpoint writes; repeat calls are diagnostic transitions.
        records.append(record)
        print(json.dumps({"smoke_q":q,"passed":True}),flush=True)
    write(ROOT/"smoke.json",{"status":"passed","records":records,
        "executed_update_calls":12,"diagnostic_transitions":12*32768,"provenance":provenance()})

def endpoint_evaluations(seed):
    c,net,r,teacher,l,mb=templates(seed)
    del teacher,mb
    for branch in ("fixed_continue","mutant_adapt"):
        out=ROOT/"endpoints"/f"{branch}_seed_{seed}.json"
        if out.exists():
            continue
        path=SOURCE/f"adaptation/{branch}_seed_{seed}/checkpoints/adapt_4096"
        state_hash=sha(path/"state.msgpack")
        runner,leo=load_dual_checkpoint(path,r,l)
        before=digest(runner,leo)
        result=evaluate(net,runner,c,4096)
        historic=json.loads((path/"evaluation.json").read_text())
        for kernel in ("fixed","mutant"):
            for selection in ("mode","sample"):
                old_records=[e for e in historic[kernel][selection]["episode_records"]
                             if selection=="mode" or e["repeat"]<8]
                new_records=result[kernel][selection]["episode_records"]
                assert len(new_records)==len(old_records)
                for a,b in zip(old_records,new_records):
                    assert all(a[k]==b[k] for k in a),(seed,branch,kernel,selection)
        assert before==digest(runner,leo) and state_hash==sha(path/"state.msgpack")
        result["source_state_sha256"]=state_hash
        result["original_episode_parity"]=True
        write(out,result)

def checkpoint_valid(path,seed,q):
    try:
        m=json.loads((path/"metadata.json").read_text())
        return (m.get("mixture_probability")==q and m.get("seed")==seed
                and m.get("execution_sha")==provenance()["execution_sha"]
                and m["state_sha256"]==sha(path/"state.msgpack"))
    except (OSError,ValueError,KeyError):
        return False

def save_checkpoint(path,r,l,c,q):
    for value in jax.tree.leaves((r,l)):
        value=np.asarray(value)
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise RuntimeError("Nonfinite checkpoint state")
    capacity("before_checkpoint_"+path.parent.parent.name+"_"+path.name)
    pending=path.with_name(path.name+f".pending.{os.getpid()}")
    if pending.exists() or path.exists():
        raise RuntimeError("Never overwrite checkpoint state")
    save_dual_checkpoint(pending,r,l,gc.spatial_wall_pass_gc_config_payload(c))
    m=json.loads((pending/"metadata.json").read_text())
    m.update({"seed":c.seed,"mixture_probability":q,"mixture":mixture_metrics(r),
              "execution_sha":provenance()["execution_sha"],"state_sha256":sha(pending/"state.msgpack"),
              "mixture_schema":"episode_kernel_rng_and_counters_v1"})
    write(pending/"metadata.json",m)
    pending.rename(path)
    write(path.parent.parent/"latest.json",{"checkpoint":str(path),
        "state_sha256":m["state_sha256"],"global_update":int(r.global_update)})

def run_cell(seed,q):
    if seed not in SEEDS or q not in PROBS:
        raise ValueError("Unregistered cell")
    cell=ROOT/cell_name(seed,q); cell.mkdir(parents=True,exist_ok=True)
    c,net,r,teacher,l,mb=templates(seed)
    mix=attach_mixture(r,seed,q)
    origin={"source_state_sha256":sha(source(seed)/"state.msgpack"),
        "commanded_base_state_sha256":digest(r,l),"q":q,"seed":seed,
        "mixture_initial_keys":np.asarray(mix.mixture.keys).tolist(),
        "execution_sha":provenance()["execution_sha"]}
    old._write_or_check(cell/"origin.json",origin,"mixture branch origin")
    checkpoint_root=cell/"checkpoints"; checkpoint_root.mkdir(exist_ok=True)
    present=[u for u in UPDATES if (checkpoint_root/f"adapt_{u}").exists()]
    for u in present:
        if not checkpoint_valid(checkpoint_root/f"adapt_{u}",seed,q):
            raise RuntimeError("Invalid checkpoint; refuse to silently roll back")
    done=max(present,default=0)
    if present:
        mix,l=load_dual_checkpoint(checkpoint_root/f"adapt_{done}",mix,l)
    assert int(mix.global_update)==512+done
    def checkpoint_and_eval(u):
        path=checkpoint_root/f"adapt_{u}"
        if not path.exists():
            save_checkpoint(path,mix,l,c,q)
        if not (path/"evaluation.json").exists():
            teacher_before=serialization.to_bytes(l)
            e=evaluate(net,mix,c,u)
            assert serialization.to_bytes(l)==teacher_before
            e["mixture_exposure"]=mixture_metrics(mix)
            write(path/"evaluation.json",e)
    checkpoint_and_eval(done)
    update=jax.jit(make_dual_leo_update(net,teacher,c,make_mixed_outcome(q),gc._batch_inputs,mb))
    for u in range(done+1,4097):
        start=time.monotonic()
        mix,l,metrics=update(mix,l)
        jax.block_until_ready(mix.global_update)
        host=old._host_metrics(metrics,u,time.monotonic()-start)
        if not all(np.all(np.isfinite(np.asarray(v))) for v in metrics.values()):
            raise RuntimeError("Nonfinite training metric")
        if u%32==0:
            host["mixture_exposure"]=mixture_metrics(mix)
            with (cell/"updates.jsonl").open("a") as f:
                f.write(json.dumps(host)+"\n")
            print(json.dumps({"cell":cell.name,"update":u,"of":4096,
                "seconds_per_update":host["seconds"],"exposure":host["mixture_exposure"]}),flush=True)
        if u%64==0:
            capacity(cell.name+"_"+str(u))
        if u in UPDATES:
            checkpoint_and_eval(u)
    assert int(mix.mixture.transitions.sum())==134217728
    assert int(mix.mixture.completed.sum())==4096*512
    write(cell/"summary.json",{"status":"complete","seed":seed,"q":q,
        "updates":4096,"transitions":134217728,"mixture_exposure":mixture_metrics(mix),
        "final_evaluation":json.loads((checkpoint_root/"adapt_4096/evaluation.json").read_text())})

def queue():
    p=provenance()
    old._write_or_check(ROOT/"run_manifest.json",p,"locked execution")
    smoke_record=json.loads((ROOT/"smoke.json").read_text())
    assert smoke_record["status"]=="passed" and smoke_record["provenance"]==p
    capacity("queue_start")
    for seed in SEEDS:
        command=[sys.executable,__file__,"--endpoints",str(seed)]
        with (ROOT/f"endpoints_seed_{seed}.log").open("a") as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        for q in PROBS:
            command=[sys.executable,__file__,"--cell",str(seed),str(q)]
            with (ROOT/f"{cell_name(seed,q)}.log").open("a") as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    cells={cell_name(s,q):json.loads((ROOT/cell_name(s,q)/"summary.json").read_text()) for s,q in jobs()}
    assert all(c["status"]=="complete" for c in cells.values())
    capacity("complete")
    write(ROOT/"result.json",{"status":"complete","jobs_completed":15,
        "training_transitions":sum(c["transitions"] for c in cells.values()),
        "completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"cells":cells,
        "execution_sha":p["execution_sha"],"diagnostic_transitions":smoke_record["diagnostic_transitions"],
        "evaluation_episode_budget":37440,"maximum_evaluation_physical_transitions":2396160})

def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke",action="store_true")
    group.add_argument("--queue",action="store_true")
    group.add_argument("--cell",nargs=2)
    group.add_argument("--endpoints",type=int)
    args=parser.parse_args()
    if os.environ.get("HACKRL_DEVICE")!="cuda" or not any(d.platform=="gpu" for d in jax.devices()):
        raise RuntimeError("Use registered GPU entrypoint")
    ROOT.mkdir(parents=True,exist_ok=True)
    provenance(); capacity("process_start")
    if args.smoke: smoke()
    elif args.queue: queue()
    elif args.endpoints is not None: endpoint_evaluations(args.endpoints)
    else: run_cell(int(args.cell[0]),float(args.cell[1]))

if __name__=="__main__":
    main()
