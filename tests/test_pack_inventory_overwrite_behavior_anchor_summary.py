import importlib.util
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_pack_inventory_overwrite_behavior_anchor_ppo",
    REPOSITORY
    / "scripts/summarize_pack_inventory_overwrite_behavior_anchor_ppo.py",
)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


def test_sign_counts_materializes_one_shot_iterables():
    values = (value for value in (1.0, -1.0, 0.0, 2.0, -2.0))
    assert SUMMARY.sign_counts(values) == {
        "positive": 2,
        "equal_within_tolerance": 1,
        "negative": 2,
    }
