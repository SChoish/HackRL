"""Engine replay for the CRAFT-REMAIN design. No training."""

from hackrl.craft_remain import (
    GROWTH_PERIOD,
    defect_script,
    normal_script,
    physical_total,
    replay,
    ripe_start,
    shortest_delivery,
    step,
    unripe_start,
)


def test_normal_craft_is_the_trigger_on_the_mutant_only():
    fixed = replay(["DO", "DO", "FILL_A", "FILL_B", "CRAFT"], mutant=False, start=(9, 8, 0, 0, 0, 0, 0, 0, 2, 0, 0))
    mutant = replay(["DO", "DO", "FILL_A", "FILL_B", "CRAFT"], mutant=True, start=(9, 8, 0, 0, 0, 0, 0, 0, 2, 0, 0))
    assert fixed["triggered"] is False
    assert fixed["state"][4:7] == (0, 0, 1)
    assert physical_total(fixed["state"]) == physical_total((9, 8, 0, 0, 0, 0, 0, 0, 2, 0, 0))
    assert mutant["triggered"] is True
    assert mutant["state"][4:7] == (1, 1, 1)
    assert mutant["exploited"] is False


def test_counterexamples_match_on_both_kernels():
    starts = [
        ripe_start(),
        unripe_start(),
        (9, 8, 0, 1, 1, 0, 0, 0, 0, 0, 0),
        (9, 8, 0, 0, 1, 1, 1, 0, 0, 0, 0),
    ]
    actions = ["DO", "CRAFT", "TAKE_A", "TAKE_B", "TAKE_OUTPUT", "DELIVER", "NOOP"]
    for start in starts:
        for action in actions:
            fixed, fixed_info = step(start, action, False)
            mutant, mutant_info = step(start, action, True)
            if action != "CRAFT":
                assert fixed == mutant
                assert fixed_info == mutant_info


def test_scripted_paths_separate_trigger_from_exploitation():
    normal_fixed = replay(normal_script(), mutant=False)
    normal_mutant = replay(normal_script(), mutant=True)
    defect_fixed = replay(defect_script(), mutant=False)
    defect_mutant = replay(defect_script(), mutant=True)
    assert normal_fixed["delivered"] == 3 and normal_fixed["length"] == 2 * GROWTH_PERIOD + 5
    assert normal_mutant["delivered"] == 3 and normal_mutant["triggered"] is False
    assert defect_fixed["delivered"] == 2 and defect_fixed["triggered"] is False
    assert defect_mutant["triggered"] is True
    assert defect_mutant["retained_taken"] == 2
    assert defect_mutant["exploited"] is True
    assert defect_mutant["delivered"] >= 3
    assert defect_mutant["length"] < normal_fixed["length"]


def test_shortest_normal_path_is_slower_than_the_mutant_path():
    fixed = shortest_delivery(False)
    mutant = shortest_delivery(True)
    assert fixed["length"] == 37
    assert fixed["actions"] == normal_script()
    assert fixed["triggered"] is False
    assert mutant["length"] == 24
    assert mutant["exploited"] is True
    assert mutant["retained_taken"] == 1
    assert mutant["length"] < fixed["length"]
    fixed_result = replay(mutant["actions"], mutant=False)
    mutant_result = replay(mutant["actions"], mutant=True)
    assert fixed_result["delivered"] == 2
    assert mutant_result["delivered"] == 3
    assert mutant_result["triggered"] is True
