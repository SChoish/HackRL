from hackrl.tick_claim import TickClaimVariant

from eval_tick_claim_reservation_probe import (
    build_reservation_probe_states,
    probe_environment_contract,
)


def test_probe_states_are_the_three_legal_timings():
    states = build_reservation_probe_states()
    bare = states["no_reservation"]
    nxt = states["due_next"]
    now = states["due_now"]
    assert bool(bare.crop_ripe) and not bool(bare.reservation_present)
    assert bool(nxt.reservation_present)
    assert int(nxt.reservation_due_tick) == int(nxt.tick) + 1
    assert int(now.reservation_due_tick) == int(now.tick)
    assert probe_environment_contract(states) == {
        "no_reservation": {"fixed": 1, "mutant": 1},
        "due_next": {"fixed": 1, "mutant": 1},
        "due_now": {"fixed": 1, "mutant": 2},
    }
    assert TickClaimVariant.MUTANT.value == "mutant"
