import jax
import jax.numpy as jnp
import pytest
from craftax.craftax_classic.constants import BlockType
from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax_classic.world_gen import generate_world


@pytest.fixture(scope="session")
def static_params():
    return StaticEnvParams()


@pytest.fixture()
def base_state(static_params):
    params = EnvParams(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
    )
    state = generate_world(jax.random.PRNGKey(0), params, static_params)
    state = state.replace(
        map=jnp.full_like(state.map, BlockType.GRASS.value),
        mob_map=jnp.zeros_like(state.mob_map),
        player_position=jnp.array([8, 8], dtype=jnp.int32),
        player_health=jnp.asarray(9),
        player_food=jnp.asarray(9),
        player_drink=jnp.asarray(9),
        player_energy=jnp.asarray(9),
        is_sleeping=jnp.asarray(False),
        zombies=state.zombies.replace(mask=jnp.zeros_like(state.zombies.mask)),
        cows=state.cows.replace(mask=jnp.zeros_like(state.cows.mask)),
        skeletons=state.skeletons.replace(mask=jnp.zeros_like(state.skeletons.mask)),
        arrows=state.arrows.replace(mask=jnp.zeros_like(state.arrows.mask)),
        growing_plants_mask=jnp.zeros_like(state.growing_plants_mask),
        growing_plants_age=jnp.zeros_like(state.growing_plants_age),
        achievements=jnp.zeros_like(state.achievements),
        timestep=jnp.asarray(0),
    )
    return state

