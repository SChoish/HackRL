"""Deterministic paths used to validate the Easy task fixtures."""

from craftax.craftax_classic.constants import Action

from hackrl.tasks import EasyTask


EASY_NORMAL_PATHS = {
    EasyTask.R_E: (
        Action.UP.value,
        Action.DO.value,
        Action.MAKE_IRON_PICKAXE.value,
    ),
    EasyTask.B_E: (Action.DOWN.value,) * 15 + (Action.DO.value,),
    EasyTask.L_E: (
        Action.DO.value,
        Action.RIGHT.value,
        Action.DO.value,
    ),
}

EASY_EXPLOIT_PATHS = {
    EasyTask.R_E: (Action.MAKE_IRON_PICKAXE.value,),
    EasyTask.B_E: (Action.DO.value,),
    EasyTask.L_E: (Action.DO.value, Action.DO.value),
}

# Waste the starting wood, mine a detour tree, then follow the original
# 3-step iron path. Used only with FixtureVersion.R_E_REPLENISH.
R_E_WOOD_WASTE_RECOVERY_PATH = (
    Action.MAKE_WOOD_PICKAXE.value,
    Action.DOWN.value,
    Action.RIGHT.value,
    Action.DO.value,
    Action.LEFT.value,
    Action.UP.value,
    Action.UP.value,
    Action.DO.value,
    Action.MAKE_IRON_PICKAXE.value,
)

# Shared post-craft diamond segment. Both paths craft at (8, 8), then
# DOWN, DO mines the diamond at (10, 8).
R_M_DIAMOND_SEGMENT = (
    Action.DOWN.value,
    Action.DO.value,
)
R_M_NORMAL_PATH = (
    Action.UP.value,
    Action.DO.value,
    Action.DOWN.value,
    Action.MAKE_IRON_PICKAXE.value,
) + R_M_DIAMOND_SEGMENT
R_M_EXPLOIT_PATH = (Action.MAKE_IRON_PICKAXE.value,) + R_M_DIAMOND_SEGMENT
