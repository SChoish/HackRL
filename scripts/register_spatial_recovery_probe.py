"""Atomically register the evaluation with the existing host watcher."""
import fcntl
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0,'/home/ext_csv/server_management/scripts/hosts/beyondg')
from register_queue import register
from queue_worker import atomic_json
from watch_beyondg_lease import load_config

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/raid/ext_csv/HackRL/runs/spatial_recovery_probe_v1')
CONFIG=Path('/home/ext_csv/server_management/config/ext_csv.json')
SCRIPTS=['scripts/run_spatial_recovery_probe_gpu.sh','scripts/run_spatial_recovery_probe.py',
         'scripts/guard_identity_checked_gpu_keepalive.py']


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with CONFIG.with_suffix('.json.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before=load_config(path=CONFIG)
        with tempfile.TemporaryDirectory(prefix='hackrl-probe-register-') as temp:
            staging=Path(temp)/'config.json';shutil.copy2(CONFIG,staging)
            cfg=register(staging,REPO/'scripts/run_spatial_recovery_probe_cpu.sh',
                REPO/'scripts/run_spatial_recovery_probe_gpu.sh',REPO,[REPO/p for p in SCRIPTS])
            cfg['worker']['process_patterns']+=SCRIPTS
            atomic_json(staging,cfg);load_config(path=staging)
            if not (ROOT/'watcher_before.json').exists(): atomic_json(ROOT/'watcher_before.json',before)
            atomic_json(CONFIG,cfg)
    print('Registered frozen evaluation. Host adoption must be verified in lease.json.')


if __name__=='__main__':main()
