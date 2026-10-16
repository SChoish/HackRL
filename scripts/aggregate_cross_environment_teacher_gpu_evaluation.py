#!/usr/bin/env python3
"""Aggregate the frozen common teacher-matrix GPU evaluation.

The aggregation keeps kernel, policy view, and start family separate.  Seed
means are unweighted because every seed sees the same frozen panel within a
cell.  Task-level means are never pooled across panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path(
    "/raid/ext_csv/HackRL/evaluations/cross_environment_teacher_matrix_v1_gpu"
)
DEFAULT_JSON = Path(
    "docs/manifests/cross_environment_teacher_matrix_v1_results.json"
)
DEFAULT_REPORT = Path("docs/cross_environment_teacher_matrix_v1_results_20261016_ko.md")
METRICS = (
    "success_rate",
    "exploit_rate",
    "trigger_rate",
    "mean_discounted_return",
)
CONTRASTS = (
    ("D-task", "D-frozen"),
    ("D-on", "D-frozen"),
    ("D-on", "D-off"),
    ("D-on", "D-task"),
)


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _distribution(values: list[float]) -> dict:
    if not values:
        raise ValueError("empty distribution")
    return {
        "mean": statistics.mean(values),
        "seed_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "positive_seeds": sum(value > 0 for value in values),
        "zero_seeds": sum(value == 0 for value in values),
        "negative_seeds": sum(value < 0 for value in values),
    }


def _group_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    keys = (
        "environment",
        "condition",
        "trained_variant",
        "checkpoint_origin",
        "kernel",
        "policy_view",
        "family",
    )
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, members in sorted(grouped.items()):
        seeds = [int(row["seed"]) for row in members]
        if len(seeds) != len(set(seeds)):
            raise RuntimeError(f"duplicate seed in aggregate group: {key}")
        output.append(
            {
                **dict(zip(keys, key)),
                "seeds": sorted(seeds),
                "seed_count": len(seeds),
                "metrics": {
                    metric: _distribution([float(row[metric]) for row in members])
                    for metric in METRICS
                },
            }
        )
    return output


def _paired_adaptation_effects(rows: list[dict]) -> list[dict]:
    """Mutant-trained minus fixed-trained, both evaluated in mutant kernel."""
    cells: dict[tuple, dict[str, dict[int, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row["condition"] == "source" or row["kernel"] != "mutant":
            continue
        key = (
            row["environment"],
            row["condition"],
            row["policy_view"],
            row["family"],
        )
        cells[key][row["trained_variant"]][int(row["seed"])] = row
    output = []
    for key, branches in sorted(cells.items()):
        fixed = branches.get("fixed", {})
        mutant = branches.get("mutant", {})
        seeds = sorted(set(fixed) & set(mutant))
        if not seeds or set(fixed) != set(mutant):
            raise RuntimeError(f"unpaired fixed/mutant branches: {key}")
        per_seed = []
        for seed in seeds:
            per_seed.append(
                {
                    "seed": seed,
                    **{
                        metric: float(mutant[seed][metric])
                        - float(fixed[seed][metric])
                        for metric in METRICS
                    },
                }
            )
        output.append(
            {
                "environment": key[0],
                "condition": key[1],
                "policy_view": key[2],
                "family": key[3],
                "evaluation_kernel": "mutant",
                "estimand": "mutant_trained_minus_fixed_trained",
                "seeds": seeds,
                "per_seed": per_seed,
                "metrics": {
                    metric: _distribution([row[metric] for row in per_seed])
                    for metric in METRICS
                },
            }
        )
    return output


def _contrast_evidence(environment: str, left: str, right: str) -> dict:
    pair = {left, right}
    if environment == "pack_restore" and pair == {"D-on", "D-off"}:
        return {
            "class": "historical_equivalence_verified",
            "single_factor": "policy imitation during adaptation",
        }
    if environment == "tick_claim" and pair == {"D-task", "D-frozen"}:
        return {
            "class": "new_same_code_and_initialization",
            "single_factor": None,
            "note": "Teacher updating and its target set differ from a frozen teacher.",
        }
    if environment == "craft_remain" and pair <= {
        "D-task",
        "D-frozen",
        "D-off",
    }:
        return {
            "class": "new_same_code_and_initialization",
            "single_factor": None,
            "note": "Conditions share code and initialization but can differ in more than one teacher/BC toggle.",
        }
    return {
        "class": "exploratory_due_to_unresolved_code_difference",
        "single_factor": None,
    }


def _condition_contrasts(effects: list[dict]) -> list[dict]:
    lookup = {
        (
            row["environment"],
            row["condition"],
            row["policy_view"],
            row["family"],
        ): row
        for row in effects
    }
    output = []
    environments = sorted({key[0] for key in lookup})
    panels = sorted({(key[2], key[3]) for key in lookup})
    for environment in environments:
        for view, family in panels:
            for left, right in CONTRASTS:
                left_row = lookup.get((environment, left, view, family))
                right_row = lookup.get((environment, right, view, family))
                if left_row is None or right_row is None:
                    continue
                left_seeds = {row["seed"]: row for row in left_row["per_seed"]}
                right_seeds = {row["seed"]: row for row in right_row["per_seed"]}
                seeds = sorted(set(left_seeds) & set(right_seeds))
                if set(left_seeds) != set(right_seeds):
                    raise RuntimeError(
                        f"unpaired condition contrast: {environment} {left} {right}"
                    )
                per_seed = [
                    {
                        "seed": seed,
                        **{
                            metric: left_seeds[seed][metric]
                            - right_seeds[seed][metric]
                            for metric in METRICS
                        },
                    }
                    for seed in seeds
                ]
                output.append(
                    {
                        "environment": environment,
                        "left_condition": left,
                        "right_condition": right,
                        "policy_view": view,
                        "family": family,
                        "estimand": (
                            "difference_of_mutant_vs_fixed_training_effects: "
                            f"{left}_minus_{right}"
                        ),
                        "evidence": _contrast_evidence(environment, left, right),
                        "seeds": seeds,
                        "per_seed": per_seed,
                        "metrics": {
                            metric: _distribution([row[metric] for row in per_seed])
                            for metric in METRICS
                        },
                    }
                )
    return output


def _load_rows(root: Path) -> tuple[dict, dict, list[dict]]:
    manifest_path = root / "evaluation_manifest.json"
    execution_path = root / "execution_receipt.json"
    manifest = json.loads(manifest_path.read_text())
    execution = json.loads(execution_path.read_text())
    if not (
        execution.get("passed") is True
        and execution.get("completed_tasks") == 1440
        and execution.get("episodes") == 88200
        and execution.get("independent_replays_passed") == 1440
        and execution.get("evaluation_execution_commit")
        == manifest.get("evaluation_execution_commit")
        and execution.get("matrix_sha256") == manifest.get("matrix_sha256")
        and execution.get("panel_sha256") == manifest.get("panel_sha256")
    ):
        raise RuntimeError("common evaluation execution receipt is incomplete or drifted")
    policy_lookup = {policy["id"]: policy for policy in manifest["policies"]}
    receipt_paths = sorted((root / "task_receipts").glob("*.json"))
    if len(receipt_paths) != 1440:
        raise RuntimeError(f"expected 1440 task receipts, found {len(receipt_paths)}")
    rows = []
    for receipt_path in receipt_paths:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "passed":
            raise RuntimeError(f"non-passing task receipt: {receipt_path}")
        policy = policy_lookup.get(receipt.get("policy"))
        if policy is None:
            raise RuntimeError(f"unknown policy in receipt: {receipt_path}")
        artifact = receipt["artifacts"]["summary"]
        summary_path = Path(artifact["path"])
        if (
            not summary_path.is_file()
            or summary_path.stat().st_size != int(artifact["bytes"])
            or _sha256(summary_path) != artifact["sha256"]
        ):
            raise RuntimeError(f"summary artifact drift: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if not (
            summary.get("scientific_performance_result") is True
            and summary.get("policy_id") == policy["id"]
            and summary.get("evaluation_backend")
            == manifest.get("evaluation_backend")
        ):
            raise RuntimeError(f"summary provenance drift: {summary_path}")
        rows.append(
            {
                "policy_id": policy["id"],
                "environment": policy["environment"],
                "condition": policy["condition"],
                "trained_variant": policy["trained_variant"] or "source",
                "seed": int(policy["seed"]),
                "checkpoint_origin": policy["checkpoint_origin"],
                "kernel": summary["kernel"],
                "policy_view": summary["policy_view"],
                "family": summary["family"],
                "episodes": int(summary["episodes"]),
                "steps": int(summary["steps"]),
                **{metric: float(summary[metric]) for metric in METRICS},
                "task_receipt": str(receipt_path),
                "summary": str(summary_path),
            }
        )
    return manifest, execution, rows


def aggregate(root: Path) -> dict:
    manifest, execution, rows = _load_rows(root)
    effects = _paired_adaptation_effects(rows)
    return {
        "schema_version": "hackrl_cross_environment_teacher_matrix_v1_results_v1",
        "recorded_utc": _utc(),
        "status": "complete",
        "aggregation_contract": {
            "seed_weighting": "equal within each environment/condition/branch/panel cell",
            "panel_pooling": "none; kernel, policy view, and start family remain separate",
            "adaptation_effect": (
                "mutant-trained minus fixed-trained policy, evaluated on the mutant kernel"
            ),
            "condition_contrast": "difference between paired adaptation effects",
            "seed_sd": "sample standard deviation across frozen learner seeds",
            "legacy_metrics_recomputed": False,
        },
        "provenance": {
            "evaluation_root": str(root),
            "evaluation_manifest": {
                "path": str(root / "evaluation_manifest.json"),
                "sha256": _sha256(root / "evaluation_manifest.json"),
            },
            "execution_receipt": {
                "path": str(root / "execution_receipt.json"),
                "sha256": _sha256(root / "execution_receipt.json"),
            },
            "evaluation_execution_commit": manifest["evaluation_execution_commit"],
            "checkpoint_execution_commit": execution["checkpoint_execution_commit"],
            "matrix_sha256": manifest["matrix_sha256"],
            "panel_sha256": manifest["panel_sha256"],
            "evaluation_backend": manifest["evaluation_backend"],
            "tasks": execution["tasks"],
            "episodes": execution["episodes"],
            "steps": execution["steps"],
            "independent_replays_passed": execution[
                "independent_replays_passed"
            ],
        },
        "per_policy_panel": rows,
        "group_summary": _group_summary(rows),
        "paired_adaptation_effects": effects,
        "condition_contrasts": _condition_contrasts(effects),
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _pp(value: float) -> str:
    return f"{100.0 * value:+.2f}pp"


def _find_group(result: dict, **wanted: object) -> dict:
    matches = [
        row
        for row in result["group_summary"]
        if all(row.get(key) == value for key, value in wanted.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"group lookup returned {len(matches)} rows: {wanted}")
    return matches[0]


def _find_effect(result: dict, **wanted: object) -> dict:
    matches = [
        row
        for row in result["paired_adaptation_effects"]
        if all(row.get(key) == value for key, value in wanted.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"effect lookup returned {len(matches)} rows: {wanted}")
    return matches[0]


def _find_contrast(result: dict, **wanted: object) -> dict:
    matches = [
        row
        for row in result["condition_contrasts"]
        if all(row.get(key) == value for key, value in wanted.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"contrast lookup returned {len(matches)} rows: {wanted}")
    return matches[0]


def render_report(result: dict) -> str:
    labels = {
        "tick_claim": "TICK-CLAIM",
        "pack_restore": "PACK-RESTORE",
        "craft_remain": "CRAFT-REMAIN",
    }
    opportunity = {
        "tick_claim": "common_setup",
        "pack_restore": "common_setup",
        "craft_remain": "path_check",
    }
    primary_contrasts = [
        ("tick_claim", "D-task", "D-frozen"),
        ("craft_remain", "D-task", "D-frozen"),
        ("pack_restore", "D-on", "D-off"),
    ]
    exploratory_goal_contrasts = [
        ("tick_claim", "D-on", "D-task"),
        ("craft_remain", "D-on", "D-task"),
        ("pack_restore", "D-on", "D-task"),
    ]
    contrast_cache = {}
    for environment, left, right in primary_contrasts + exploratory_goal_contrasts:
        contrast_cache[(environment, left, right, "natural")] = _find_contrast(
            result,
            environment=environment,
            left_condition=left,
            right_condition=right,
            policy_view="sample",
            family="natural_reset",
        )
        contrast_cache[(environment, left, right, "opportunity")] = _find_contrast(
            result,
            environment=environment,
            left_condition=left,
            right_condition=right,
            policy_view="sample",
            family=opportunity[environment],
        )
    tick_teacher = contrast_cache[("tick_claim", "D-task", "D-frozen", "natural")]
    tick_teacher_opp = contrast_cache[("tick_claim", "D-task", "D-frozen", "opportunity")]
    craft_teacher = contrast_cache[("craft_remain", "D-task", "D-frozen", "natural")]
    craft_teacher_opp = contrast_cache[("craft_remain", "D-task", "D-frozen", "opportunity")]
    pack_imitation = contrast_cache[("pack_restore", "D-on", "D-off", "natural")]
    pack_imitation_opp = contrast_cache[("pack_restore", "D-on", "D-off", "opportunity")]

    def group_metric(environment, condition, trained_variant, kernel, view, family, metric):
        return _find_group(
            result,
            environment=environment,
            condition=condition,
            trained_variant=trained_variant,
            kernel=kernel,
            policy_view=view,
            family=family,
        )["metrics"][metric]["mean"]

    manuscript_lines = [
        "## 원고용 핵심 결과",
        "",
        "> TICK·CRAFT에서는 모방을 유지한 task teacher 갱신 대 동결의 `ΔU`가 양수였고, PACK에서는 전체 teacher 갱신 중 모방 유지 대 중단의 `ΔU`가 양수였다. 두 비교는 서로 다른 개입이며 learner seed별 결과가 달랐다. 활용 채택의 증가는 정상 수행 보존을 보장하지 않았다.",
        "",
        "아래 주 효과는 직접 활용률 차이가 아니라 seed별 `ΔU_s = U_{s,left} − U_{s,right}`의 평균 `mean(ΔU_s)`다. `U_{s,c}`는 같은 seed·조건에서 mutant 학습 정책과 fixed 계속학습 정책을 모두 mutant kernel에 평가한 활용률 차이다. `ΔΔJ`도 같은 차이-중-차이 구조의 할인 return이며 환경 사이의 절대 크기를 직접 비교하지 않는다. 정상 성능은 각 조건의 최종 mutant 학습 정책을 fixed kernel·mode·자연 시작에서 평가한 좌/우 값이다.",
        "",
        "| 환경 | 개입: 좌 − 우 | 자연 ΔU | 기회 ΔU | 양수 seed 자연/기회 | 자연 ΔΔJ | 기회 ΔΔJ | 정상 성공률 좌/우 | 정상 return 좌/우 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for environment, left, right in primary_contrasts:
        natural = contrast_cache[(environment, left, right, "natural")]
        opp = contrast_cache[(environment, left, right, "opportunity")]
        left_success = group_metric(
            environment, left, "mutant", "fixed", "mode",
            "natural_reset", "success_rate"
        )
        right_success = group_metric(
            environment, right, "mutant", "fixed", "mode",
            "natural_reset", "success_rate"
        )
        left_return = group_metric(
            environment, left, "mutant", "fixed", "mode",
            "natural_reset", "mean_discounted_return"
        )
        right_return = group_metric(
            environment, right, "mutant", "fixed", "mode",
            "natural_reset", "mean_discounted_return"
        )
        natural_metric = natural["metrics"]["exploit_rate"]
        opp_metric = opp["metrics"]["exploit_rate"]
        manuscript_lines.append(
            f"| {labels[environment]} | {left} − {right} | "
            f"{_pp(natural_metric['mean'])} | {_pp(opp_metric['mean'])} | "
            f"{natural_metric['positive_seeds']}/{len(natural['seeds'])} · "
            f"{opp_metric['positive_seeds']}/{len(opp['seeds'])} | "
            f"{natural['metrics']['mean_discounted_return']['mean']:+.5f} | "
            f"{opp['metrics']['mean_discounted_return']['mean']:+.5f} | "
            f"{_pct(left_success)} / {_pct(right_success)} | "
            f"{left_return:.5f} / {right_return:.5f} |"
        )
    manuscript_lines.extend(
        [
            "",
            "예를 들어 PACK 자연 시작의 `ΔU`는 `(64.61−0.31)−(5.00−0.00)=59.30pp`다. 최종 mutant 학습 정책끼리의 직접 활용률 차이 `64.61−5.00=59.61pp`와 다르다.",
            "",
            "PACK `D-on − D-off`는 같은 historical run manifest·실행 코드·source checkpoint·seed·branch config를 공유하고 적응 중 모방만 달랐다는 해시·config 감사에 근거한다. 다른 run을 가로지르는 PACK 비교의 과거 코드 차이는 여전히 미해결이다.",
            "",
            "원고 그림은 [PNG](figures/cross_environment_teacher_matrix_v1_seed_effects.png)와 [PDF](figures/cross_environment_teacher_matrix_v1_seed_effects.pdf)로 제공한다. [그림 데이터 CSV](figures/cross_environment_teacher_matrix_v1_seed_effects.csv)는 모든 learner-seed 점을 보존한다.",
            "",
            "**그림 캡션.** 환경별 사전 지정 비교의 learner-seed 분포. TICK·CRAFT는 `D-task − D-frozen`, PACK은 별개의 `D-on − D-off` 개입이다. (a) `ΔU`는 mutant kernel의 sample 평가에서 조건별 mutant-vs-fixed 학습 효과를 다시 뺀 값이며 자연 시작과 합법적 기회 시작을 분리한다. 직접 활용률은 이 패널에 표시하지 않는다. (b) 두 조건의 최종 mutant 학습 정책을 fixed kernel의 mode·자연 시작에서 평가한 성공률이다. (c) (a)와 같은 차이-중-차이의 할인 return이다. 옅은 선은 같은 learner seed를 잇고, seed별 수평 jitter는 값이 겹쳐도 각 점을 보이게 한다. 검은 마름모는 seed 동일가중 평균이다. 88,200개 평가 episode는 각 정책 추정의 정밀도를 높이지만 독립 learner 수를 늘리지 않는다.",
            "",
            "**Manuscript text.** In TICK and CRAFT, the prespecified `D-task − D-frozen` difference-in-differences in exploit adoption was positive on average: +11.56/+25.16 pp and +60.00/+60.00 pp for natural/opportunity starts, respectively. In PACK, the distinct prespecified `D-on − D-off` contrast was +59.30/+59.06 pp. These descriptive contrasts were concentrated in subsets of learner seeds, and no significance or equivalence test was performed. `D-on − D-task` comparisons were exploratory because of unresolved historical code differences and do not identify an effect of full-goal updating. Increased exploit adoption did not guarantee normal-task preservation.",
            "",
        ]
    )

    lines = [
        "# 공통 Teacher Matrix v1 결과",
        "",
        "이 문서는 [결과 JSON](manifests/cross_environment_teacher_matrix_v1_results.json)의 기계 생성 요약이다. 전체 seed별 수치와 artifact 경로는 JSON을 정본으로 삼는다. 실험 계약과 비교 근거는 [matrix 명세](manifests/cross_environment_teacher_matrix_v1.json)와 [evidence manifest](manifests/cross_environment_teacher_matrix_v1_evidence.json)에 있다.",
        "",
        "## 실행 무결성",
        "",
        f"- 정책 180개, task {result['provenance']['tasks']:,}개, episode {result['provenance']['episodes']:,}개, step {result['provenance']['steps']:,}개를 완료했다.",
        f"- 독립 trace replay는 {result['provenance']['independent_replays_passed']:,}/{result['provenance']['tasks']:,}개가 통과했다.",
        f"- 평가 commit은 `{result['provenance']['evaluation_execution_commit']}`, 학습 실행 commit은 `{result['provenance']['checkpoint_execution_commit']}`다.",
        "- 1,440개 task는 180개 정책 × 정상/결함 kernel × mode/sample × 자연/기회 시작의 2×2×2 완전 조합이다. TICK·PACK은 seed별 mode 32회와 sample 128회, CRAFT는 mode 2회와 sample 8회를 각 시작 family에서 실행했다.",
        "- replay는 저장된 전후 상태에 별도 자원 보존식을 적용하며 transition kernel·actor·teacher를 호출하지 않는다. 코드 경로는 독립적이지만 같은 프로젝트와 설계자가 만든 검증기이므로 외부에서 작성된 ground truth는 아니다.",
        "- 아래 평균은 seed 동일 가중이며 kernel·mode/sample·시작 family를 합치지 않았다. 결측 cell은 없고 유의성·동등성 검정은 수행하지 않았다.",
        "",
        "## 조건과 지표",
        "",
        "- TICK-CLAIM, PACK-RESTORE, CRAFT-REMAIN은 각각 reservation-settlement, source-growth, crafting-input 자원 보존 결함 fixture다. `fixed`는 정상 전이, `mutant`는 결함 전이다.",
        "- `Source`는 적응 전 정상 사전학습 checkpoint다. `D-on`은 전체 teacher head 갱신과 정책 모방을 유지하고, `D-task`는 해당 과제 head만 갱신하며 모방을 유지한다. `D-frozen`은 teacher 전체 상태를 동결한 채 모방하고, `D-off`는 teacher 갱신을 유지하되 정책 BC 계수를 0으로 둔다.",
        "- `mode`는 greedy 행동 1회, `sample`은 확률적 행동 4회 반복이다. 자연 시작은 `natural_reset`, 합법적 기회 시작은 TICK·PACK의 `common_setup`과 CRAFT의 `path_check` 고정 패널이다.",
        "- 성공률은 최종 과제 완료 비율이다. 활용률은 TICK·PACK에서는 결함으로 생긴 자원이 실제 납품된 episode, CRAFT에서는 결함 발동 뒤 보존 자원을 회수하고 초과 납품까지 한 episode의 비율이다.",
        "- `U_{s,c}`는 seed `s`와 조건 `c`에서 mutant 학습 정책의 활용률에서 fixed 계속학습 정책의 활용률을 뺀 값이다. 조건 대비는 `ΔU_s = U_{s,left} − U_{s,right}`이고 표에는 seed 동일가중 `mean(ΔU_s)`를 보고한다. `양수 seed`는 반올림 전 `ΔU_s>0`인 seed 수다.",
        "",
        *manuscript_lines,
        "## Source와 mutant 적응 정책의 공통 패널 결과",
        "",
        "아래 활용률과 자연 성공률은 mutant kernel의 sample 평가다. 정상 성공률은 같은 정책을 fixed kernel의 mode·자연 시작에서 평가한 값이다. `D-*` 행은 mutant 환경에서 학습한 분기를 사용한다.",
        "",
        "| 환경 | 정책 | 자연 활용률 | 기회 활용률 | 자연 성공률 | 정상 kernel mode 성공률 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for environment in labels:
        source_natural = _find_group(
            result,
            environment=environment,
            condition="source",
            trained_variant="source",
            kernel="mutant",
            policy_view="sample",
            family="natural_reset",
        )
        source_opp = _find_group(
            result,
            environment=environment,
            condition="source",
            trained_variant="source",
            kernel="mutant",
            policy_view="sample",
            family=opportunity[environment],
        )
        source_normal = _find_group(
            result,
            environment=environment,
            condition="source",
            trained_variant="source",
            kernel="fixed",
            policy_view="mode",
            family="natural_reset",
        )
        lines.append(
            f"| {labels[environment]} | Source | {_pct(source_natural['metrics']['exploit_rate']['mean'])} | {_pct(source_opp['metrics']['exploit_rate']['mean'])} | {_pct(source_natural['metrics']['success_rate']['mean'])} | {_pct(source_normal['metrics']['success_rate']['mean'])} |"
        )
        for condition in ("D-on", "D-task", "D-frozen", "D-off"):
            natural = _find_group(
                result,
                environment=environment,
                condition=condition,
                trained_variant="mutant",
                kernel="mutant",
                policy_view="sample",
                family="natural_reset",
            )
            opp = _find_group(
                result,
                environment=environment,
                condition=condition,
                trained_variant="mutant",
                kernel="mutant",
                policy_view="sample",
                family=opportunity[environment],
            )
            normal = _find_group(
                result,
                environment=environment,
                condition=condition,
                trained_variant="mutant",
                kernel="fixed",
                policy_view="mode",
                family="natural_reset",
            )
            lines.append(
                f"| {labels[environment]} | {condition} | {_pct(natural['metrics']['exploit_rate']['mean'])} | {_pct(opp['metrics']['exploit_rate']['mean'])} | {_pct(natural['metrics']['success_rate']['mean'])} | {_pct(normal['metrics']['success_rate']['mean'])} |"
            )
    lines.extend(
        [
            "",
            "## Mutant 경험의 활용 채택 효과",
            "",
            "화살표는 fixed 계속학습 분기에서 mutant 학습 분기로 바뀐 활용률이고, `U`는 그 paired 차이의 seed 평균이다. 모두 mutant kernel의 sample 평가다.",
            "",
            "| 환경 | 조건 | 자연 fixed→mutant | 자연 U | 양수 seed | 기회 fixed→mutant | 기회 U | 양수 seed |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for environment in labels:
        for condition in ("D-on", "D-task", "D-frozen", "D-off"):
            natural = _find_effect(
                result,
                environment=environment,
                condition=condition,
                policy_view="sample",
                family="natural_reset",
            )
            opp = _find_effect(
                result,
                environment=environment,
                condition=condition,
                policy_view="sample",
                family=opportunity[environment],
            )
            n_metric = natural["metrics"]["exploit_rate"]
            o_metric = opp["metrics"]["exploit_rate"]
            fixed_natural = group_metric(
                environment, condition, "fixed", "mutant", "sample",
                "natural_reset", "exploit_rate"
            )
            mutant_natural = group_metric(
                environment, condition, "mutant", "mutant", "sample",
                "natural_reset", "exploit_rate"
            )
            fixed_opp = group_metric(
                environment, condition, "fixed", "mutant", "sample",
                opportunity[environment], "exploit_rate"
            )
            mutant_opp = group_metric(
                environment, condition, "mutant", "mutant", "sample",
                opportunity[environment], "exploit_rate"
            )
            lines.append(
                f"| {labels[environment]} | {condition} | {_pct(fixed_natural)}→{_pct(mutant_natural)} | {_pp(n_metric['mean'])} | {n_metric['positive_seeds']}/{len(natural['seeds'])} | {_pct(fixed_opp)}→{_pct(mutant_opp)} | {_pp(o_metric['mean'])} | {o_metric['positive_seeds']}/{len(opp['seeds'])} |"
            )
    lines.extend(
        [
            "",
            "## 사전 지정 비교",
            "",
            "`ΔU`는 왼쪽 조건의 paired 활용 채택 효과에서 오른쪽 조건의 효과를 뺀 값이다.",
            "",
            "| 비교 | 자연 ΔU | 기회 ΔU | 근거 상태 |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for environment, left, right in primary_contrasts + exploratory_goal_contrasts:
        natural = contrast_cache[(environment, left, right, "natural")]
        opp = contrast_cache[(environment, left, right, "opportunity")]
        lines.append(
            f"| {labels[environment]} {left} − {right} | {_pp(natural['metrics']['exploit_rate']['mean'])} | {_pp(opp['metrics']['exploit_rate']['mean'])} | `{natural['evidence']['class']}` |"
        )
    lines.extend(
        [
            "",
            "주요 matched 비교의 자연 시작 seed별 ΔU는 다음과 같다. 값은 percentage point다.",
            "",
            "| 비교 | seed별 자연 ΔU |",
            "| --- | --- |",
        ]
    )
    for environment, left, right in primary_contrasts:
        row = contrast_cache[(environment, left, right, "natural")]
        seed_values = ", ".join(
            f"{item['seed']}:{_pp(item['exploit_rate'])}" for item in row["per_seed"]
        )
        lines.append(f"| {labels[environment]} {left} − {right} | {seed_values} |")
    lines.extend(
        [
            "",
            "## 정상 수행 보존",
            "",
            "아래 값은 fixed kernel·mode·자연 시작 성공률이다. Source는 적응 전 기준선이고, 두 학습 열은 각각 fixed 계속학습과 mutant 적응의 최종 정책이다.",
            "",
            "| 환경 | 조건 | Source | fixed 학습 | mutant 학습 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for environment in labels:
        source_normal = group_metric(
            environment, "source", "source", "fixed", "mode",
            "natural_reset", "success_rate"
        )
        for condition in ("D-on", "D-task", "D-frozen", "D-off"):
            fixed_normal = group_metric(
                environment, condition, "fixed", "fixed", "mode",
                "natural_reset", "success_rate"
            )
            mutant_normal = group_metric(
                environment, condition, "mutant", "fixed", "mode",
                "natural_reset", "success_rate"
            )
            lines.append(
                f"| {labels[environment]} | {condition} | {_pct(source_normal)} | {_pct(fixed_normal)} | {_pct(mutant_normal)} |"
            )

    tick_source_natural = group_metric(
        "tick_claim", "source", "source", "mutant", "sample",
        "natural_reset", "exploit_rate"
    )
    tick_source_opp = group_metric(
        "tick_claim", "source", "source", "mutant", "sample",
        opportunity["tick_claim"], "exploit_rate"
    )
    pack_source_natural = group_metric(
        "pack_restore", "source", "source", "mutant", "sample",
        "natural_reset", "exploit_rate"
    )
    pack_source_opp = group_metric(
        "pack_restore", "source", "source", "mutant", "sample",
        opportunity["pack_restore"], "exploit_rate"
    )
    craft_source_natural = group_metric(
        "craft_remain", "source", "source", "mutant", "sample",
        "natural_reset", "exploit_rate"
    )
    craft_source_opp = group_metric(
        "craft_remain", "source", "source", "mutant", "sample",
        opportunity["craft_remain"], "exploit_rate"
    )
    tick_doff_fixed = group_metric(
        "tick_claim", "D-off", "fixed", "fixed", "mode",
        "natural_reset", "success_rate"
    )
    craft_doff_fixed = group_metric(
        "craft_remain", "D-off", "fixed", "fixed", "mode",
        "natural_reset", "success_rate"
    )
    craft_doff_mutant = group_metric(
        "craft_remain", "D-off", "mutant", "fixed", "mode",
        "natural_reset", "success_rate"
    )
    tick_positive = tick_teacher["metrics"]["exploit_rate"]["positive_seeds"]
    craft_positive = craft_teacher["metrics"]["exploit_rate"]["positive_seeds"]
    pack_distribution = pack_imitation["metrics"]["exploit_rate"]
    lines.extend(
        [
            "",
            "## 해석",
            "",
            f"- 신규 동일 코드·초기화 비교에서 task-goal teacher를 갱신한 `D-task`의 활용 채택 효과는 `D-frozen`보다 TICK 자연 {_pp(tick_teacher['metrics']['exploit_rate']['mean'])}, 기회 {_pp(tick_teacher_opp['metrics']['exploit_rate']['mean'])} 높았고, CRAFT에서는 자연 {_pp(craft_teacher['metrics']['exploit_rate']['mean'])}, 기회 {_pp(craft_teacher_opp['metrics']['exploit_rate']['mean'])} 높았다. 방향은 두 환경에서 같았지만 양의 차이는 각각 {tick_positive}/5와 {craft_positive}/5 seed에 집중됐다.",
            f"- 동등성이 확인된 PACK `D-on − D-off` 비교에서 활용 채택 효과 차이는 자연 {_pp(pack_imitation['metrics']['exploit_rate']['mean'])}, 기회 {_pp(pack_imitation_opp['metrics']['exploit_rate']['mean'])}였다. 자연 시작 seed별로 {pack_distribution['positive_seeds']}개 양수, {pack_distribution['zero_seeds']}개 0, {pack_distribution['negative_seeds']}개 음수였으므로 모든 seed에 일관된 효과는 아니다.",
            f"- 전체 목표 갱신 `D-on`과 task-only `D-task`의 탐색적 ΔU는 TICK {_pp(contrast_cache[('tick_claim', 'D-on', 'D-task', 'natural')]['metrics']['exploit_rate']['mean'])}, CRAFT {_pp(contrast_cache[('craft_remain', 'D-on', 'D-task', 'natural')]['metrics']['exploit_rate']['mean'])}, PACK {_pp(contrast_cache[('pack_restore', 'D-on', 'D-task', 'natural')]['metrics']['exploit_rate']['mean'])}였다. 과거 코드 차이가 남아 있어 전체 목표 갱신의 단일 요인 효과로 해석하지 않는다.",
            f"- 정상 수행은 조건과 학습 분기에 따라 달랐다. TICK D-off fixed 학습은 {_pct(tick_doff_fixed)}, CRAFT D-off는 fixed {_pct(craft_doff_fixed)}·mutant {_pct(craft_doff_mutant)}였다. 활용 채택과 정상 수행 보존은 별도 결과다.",
            f"- Source의 sample 활용은 TICK 자연 {_pct(tick_source_natural)}·기회 {_pct(tick_source_opp)}, PACK 자연 {_pct(pack_source_natural)}·기회 {_pct(pack_source_opp)}, CRAFT 자연 {_pct(craft_source_natural)}·기회 {_pct(craft_source_opp)}였다. 최종 활용률과 mutant 경험으로 생긴 paired 증가분 `U`를 구분해야 한다.",
            "",
            "## 주장 경계",
            "",
            "- TICK·CRAFT의 신규 조건과 historical D-on/D-off 간 비교, 그리고 서로 다른 PACK run을 가로지르는 비교는 탐색적이다.",
            "- 결과 범위는 이 세 fixture와 고정 패널이다. 같은 방향의 환경 재현은 관측됐지만 teacher가 어떤 내부 기전으로 행동을 바꿨는지는 이 평가만으로 규명하지 않는다.",
            "- seed 수는 TICK·CRAFT 5개, PACK 10개이며 seed SD와 전체 per-seed 값은 결과 JSON에 보존했다. 별도 신뢰구간이나 유의성·동등성 검정은 하지 않았다.",
            "- 과거의 좁은 opportunity 지표는 소급 변경하지 않았으며, 본 문서는 새 고정 패널의 exhaustive legal opportunity family 결과를 별도로 보고한다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = aggregate(args.evaluation_root.resolve())
    result["provenance"]["aggregation_script"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256(Path(__file__).resolve()),
    }
    report = render_report(result)
    if not args.check_only:
        _atomic_json(args.output_json, result)
        _atomic_text(args.output_report, report)
    print(
        json.dumps(
            {
                "status": result["status"],
                "per_policy_panel_rows": len(result["per_policy_panel"]),
                "group_summary_rows": len(result["group_summary"]),
                "paired_adaptation_effect_rows": len(
                    result["paired_adaptation_effects"]
                ),
                "condition_contrast_rows": len(result["condition_contrasts"]),
                "output_json": None if args.check_only else str(args.output_json),
                "output_report": None if args.check_only else str(args.output_report),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
