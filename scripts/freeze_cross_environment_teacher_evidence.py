#!/usr/bin/env python3
"""Create the non-invasive evidence index for teacher matrix v1.

This reads and hashes historical artifacts but never writes into checkpoint
trees.  Large checkpoints remain at their existing locations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/manifests/cross_environment_teacher_matrix_v1_evidence.json"
INITIAL_WORKTREE = [
    " M docs/manifests/craft_remain_ppo_compare_v1.json",
    " M docs/manifests/dual_leo_ppo_compare_v1_results.json",
    " M docs/manifests/tick_claim_gc_workshop12_base_v1_completion.json",
    " M docs/manifests/tick_claim_gc_workshop12_goal_eval_v1.json",
    " M docs/third_defect_candidates_ko.md",
    " M scripts/eval_tick_claim_gc_workshop12_goals.py",
    " M src/hackrl/tick_claim_gc_goal_eval.py",
    " M tests/test_tick_claim_gc_goal_eval.py",
    "?? .worktrees/",
    "?? 261014.pptx",
    "?? AGENTS.md",
    "?? HackRL_Research_Plan_20261014.docx",
    "?? HackRL_Research_Roadmap_KO_20261012.pdf",
    "?? c415dbb_candidate_specs.patch",
    "?? docs/cross_environment_teacher_matrix_20261015_ko.md",
    "?? docs/manifests/craft_remain_ppo_compare_v1_results.json",
    "?? docs/manifests/cross_environment_teacher_matrix_v1.json",
    "?? docs/manifests/tick_claim_line_freeze_v1.json",
    "?? ops/",
    "?? presentation_261014_v1/",
    "?? scripts/eval_tick_claim_pretrain_harvest.py",
    "?? scripts/eval_tick_claim_pretrain_paths.py",
    "?? scripts/repair_pack_restore_delivery_only_teacher_resume_logs.py",
    "?? scripts/run_pack_restore_delivery_only_teacher_resume_gpu_queue.sh",
    "?? scripts/run_pack_restore_delivery_only_teacher_watcher_cpu.sh",
    "?? scripts/run_pack_restore_delivery_only_teacher_watcher_gpu.sh",
    "?? scripts/run_pack_restore_scale_bc_watchdog.sh",
    "?? scripts/validate_cross_environment_teacher_matrix.py",
    "?? tests/test_cross_environment_teacher_matrix.py",
]


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(path: Path) -> dict:
    path = path.resolve()
    files = {}
    for name in ("state.msgpack", "config.json", "metadata.json", "arm.json"):
        item = path / name
        if item.is_file():
            files[name] = {"sha256": sha256(item), "bytes": item.stat().st_size}
    if not {"state.msgpack", "config.json", "metadata.json"}.issubset(files):
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for name, record in sorted(files.items()):
        digest.update(name.encode() + b"\0" + record["sha256"].encode() + b"\n")
    metadata = json.loads((path / "metadata.json").read_text())
    return {
        "path": str(path),
        "fingerprint_sha256": digest.hexdigest(),
        "files": files,
        "global_update": int(metadata["global_update"]),
    }


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def seed_paths(root: Path, seeds: range, branches: bool, method: str) -> tuple[list[Path], list[Path]]:
    if branches:
        finals = [root / method / variant / f"seed{seed}/checkpoints/adapt_4096" for seed in seeds for variant in ("fixed", "mutant")]
        curves = [root / method / variant / f"seed{seed}/curve/adapt_4096.json" for seed in seeds for variant in ("fixed", "mutant")]
    else:
        finals = [root / method / "pretrain" / f"seed{seed}/checkpoints/update_512" for seed in seeds]
        curves = []
    return finals, curves


def build() -> dict:
    groups = {}
    checkpoints: dict[str, dict] = {}
    artifacts: dict[str, dict] = {}

    def add_group(name, *, execution_commit, commit_status, config, sources, finals, curves, evaluation_panel, run_manifest=None, result=None):
        source_records = []
        final_records = []
        curve_records = []
        for label, paths, output in (("source", sources, source_records), ("final", finals, final_records)):
            del label
            for path in paths:
                record = checkpoint(path)
                checkpoints.setdefault(record["fingerprint_sha256"], record)
                output.append({"path": record["path"], "fingerprint_sha256": record["fingerprint_sha256"]})
        for path in curves:
            record = artifact(path)
            artifacts.setdefault(record["sha256"], record)
            curve_records.append({"path": record["path"], "sha256": record["sha256"]})
        config_record = artifact(config)
        artifacts.setdefault(config_record["sha256"], config_record)
        group = {
            "execution_commit": execution_commit,
            "execution_commit_status": commit_status,
            "config": {"path": config_record["path"], "sha256": config_record["sha256"]},
            "source_checkpoints": source_records,
            "final_checkpoints": final_records,
            "final_evaluation_artifacts": curve_records,
            "evaluation_panel": evaluation_panel,
        }
        for key, path in (("run_manifest", run_manifest), ("result", result)):
            if path is not None:
                record = artifact(path)
                artifacts.setdefault(record["sha256"], record)
                group[key] = {"path": record["path"], "sha256": record["sha256"]}
        groups[name] = group

    tick_root = ROOT / "runs/dual_leo_compare_v1/tick"
    tick_sources, _ = seed_paths(tick_root, range(20, 25), False, "dual")
    tick_on, tick_on_curves = seed_paths(tick_root, range(20, 25), True, "dual")
    add_group(
        "tick_source_and_D_on",
        execution_commit="9155a0d922bbf4a2d335c8bca5cc33a5ce0ff948",
        commit_status="best preserved post-run freeze; result records execution began from a dirty fbaef1d tree, so exact execution tree is unresolved",
        config=ROOT / "docs/manifests/dual_leo_ppo_compare_v1.json",
        sources=tick_sources,
        finals=tick_on,
        curves=tick_on_curves,
        evaluation_panel="validation layouts 0..15 x phases 0..1; natural_reset/common_setup; mode1/sample4; both kernels",
        result=ROOT / "docs/manifests/dual_leo_ppo_compare_v1_results.json",
    )
    tick_off_root = ROOT / "runs/dual_adapt_bc_off_v1/tick"
    tick_off, tick_off_curves = seed_paths(tick_off_root, range(20, 25), True, "adapt_bc_off")
    add_group(
        "tick_D_off",
        execution_commit="87e37c2",
        commit_status="historical freeze commit; exact full execution checkout not independently reconstructed",
        config=ROOT / "docs/manifests/dual_adapt_bc_off_v1.json",
        sources=tick_sources,
        finals=tick_off,
        curves=tick_off_curves,
        evaluation_panel="historical final curves; common full-trace panel remains pending",
    )

    craft_root = ROOT / "runs/craft_remain_compare_v1"
    craft_sources, _ = seed_paths(craft_root, range(20, 25), False, "dual")
    craft_on, craft_curves = seed_paths(craft_root, range(20, 25), True, "dual")
    add_group(
        "craft_source_and_D_on",
        execution_commit="fd201788bb716b4ee4ea9b8e1579c81f9d480c23",
        commit_status="runner and craft GC sources match this commit; current dual_leo.py differs and is not claimed equivalent",
        config=ROOT / "docs/manifests/craft_remain_ppo_compare_v1.json",
        sources=craft_sources,
        finals=craft_on,
        curves=craft_curves,
        evaluation_panel="natural_reset/path_check x ripe/unripe; mode1/sample4; both kernels",
        result=ROOT / "docs/manifests/craft_remain_ppo_compare_v1_results.json",
    )

    pack_root = Path("/raid/ext_csv/HackRL/runs/pack_restore_scale_bc_v1/size_s/pack")
    pack_sources, _ = seed_paths(pack_root, range(30, 40), False, "dual")
    pack_on, pack_on_curves = seed_paths(pack_root, range(30, 40), True, "dual")
    pack_off, pack_off_curves = seed_paths(pack_root, range(30, 40), True, "dual_bc_off")
    add_group(
        "pack_source_D_on_D_off",
        execution_commit="eaf603b1ab378b34bf2a788927f59446cabdde50",
        commit_status="exact execution_code_sha and scientific hashes in run_manifest",
        config=ROOT / "docs/manifests/pack_restore_scale_bc_v1.json",
        sources=pack_sources,
        finals=pack_on + pack_off,
        curves=pack_on_curves + pack_off_curves,
        evaluation_panel="validation layouts 0..15 x phases 0..1; natural_reset/common_setup; mode1/sample4; both kernels",
        run_manifest=Path("/raid/ext_csv/HackRL/runs/pack_restore_scale_bc_v1/run_manifest.json"),
    )
    for name, root, method, commit, config_name in (
        ("pack_D_task", Path("/raid/ext_csv/HackRL/runs/pack_restore_delivery_only_teacher_v1/size_s/pack"), "dual_teacher_delivery_only", "a68a948e55f2e7c7bccba6fec6a7d63bbb8f79d7", "pack_restore_delivery_only_teacher_v1.json"),
        ("pack_D_frozen", Path("/raid/ext_csv/HackRL/runs/pack_restore_pretrained_teacher_freeze_v1/size_s/pack"), "dual_teacher_frozen", "8f3b97ecc30ca70127c4b4af091fa37234ad5005", "pack_restore_pretrained_teacher_freeze_v1.json"),
    ):
        finals, curves = seed_paths(root, range(30, 40), True, method)
        run_root = root.parents[1]
        add_group(
            name,
            execution_commit=commit,
            commit_status="exact execution_code_sha and scientific hashes in run_manifest",
            config=ROOT / "docs/manifests" / config_name,
            sources=pack_sources,
            finals=finals,
            curves=curves,
            evaluation_panel="frozen manifest panel; common full-trace panel remains a separate pending evaluation",
            run_manifest=run_root / "run_manifest.json",
            result=run_root / "result.json",
        )

    current_hashes = {
        path: sha256(ROOT / path)
        for path in (
            "scripts/run_dual_leo_compare.py",
            "scripts/run_craft_remain_compare.py",
            "src/hackrl/dual_leo.py",
            "src/hackrl/craft_remain_gc.py",
        )
    }
    historical = {
        "craft_freeze_commit": "fd201788bb716b4ee4ea9b8e1579c81f9d480c23",
        "scripts/run_craft_remain_compare.py": "8a5ef6720a2846adf62b68a5004267d21328f6b7ff1b075bb6b39de4a14d3e4e",
        "src/hackrl/craft_remain_gc.py": "f3b03a8484ad63f4327aee435e298c4098d2e50dd78fc3425c02b5a92d4f5c46",
        "src/hackrl/dual_leo.py": "119f496b4bf7bbf2e0d36fb72c8059263eb2c5509881c919f17c7ee3a4678471",
    }
    return {
        "schema_version": "hackrl_cross_environment_teacher_matrix_evidence_v1",
        "recorded_utc": utc(),
        "understood_as": "Freeze only evidence referenced by matrix v1, implement resumable missing-cell execution and trace evaluation, run contract smoke only, and leave the 50-job main queue unlaunched.",
        "workspace_snapshot_before_implementation": INITIAL_WORKTREE,
        "preservation": {
            "reset_stash_delete_used": False,
            "unrelated_files_included": False,
            "large_checkpoints_copied_into_git": False,
            "checkpoint_policy": "existing absolute/relative path plus content fingerprint only",
        },
        "evidence_groups": groups,
        "checkpoint_index": checkpoints,
        "artifact_index": artifacts,
        "current_vs_historical_code": {
            "current_sha256": current_hashes,
            "craft_historical_sha256": historical,
            "resolved": [
                "current scripts/run_craft_remain_compare.py equals fd201 freeze",
                "current src/hackrl/craft_remain_gc.py equals fd201 freeze",
            ],
            "unresolved": [
                "TICK D-on execution started from a dirty fbaef1d worktree; 9155a0d is the best preserved post-run freeze, not a proof of exact checkout identity",
                "current src/hackrl/dual_leo.py differs from fd201 CRAFT execution code",
                "historical TICK/CRAFT runs have no standalone run_manifest with a complete execution source-hash set",
                "common full traces do not exist yet; historical curves remain legacy metrics and are not retroactively redefined",
            ],
        },
        "git_head_when_generated": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "groups": len(value["evidence_groups"]), "checkpoints": len(value["checkpoint_index"]), "artifacts": len(value["artifact_index"])}, sort_keys=True))


if __name__ == "__main__":
    main()
