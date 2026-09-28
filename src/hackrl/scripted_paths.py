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
