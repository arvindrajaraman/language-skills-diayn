import jax
import jax.numpy as jnp
from icecream import ic

from craftax_classic.constants import *

# OBS_SIZE = 1345
# MAP_HEIGHT = 7
# MAP_WIDTH = 9
# MAP_FEATURES = 21

# obs_feature_labels = {
#     -22: 'inv/wood',
#     -21: 'inv/stone',
#     -20: 'inv/coal',
#     -19: 'inv/iron',
#     -18: 'inv/diamond',
#     -17: 'inv/sapling',
#     -16: 'inv/wood_pickaxe',
#     -15: 'inv/stone_pickaxe',
#     -14: 'inv/iron_pickaxe',
#     -13: 'inv/wood_sword',
#     -12: 'inv/stone_sword',
#     -11: 'inv/iron_sword',
#     -10: 'int/player_health',
#     -9: 'int/player_food',
#     -8: 'int/player_drink',
#     -7: 'int/player_energy',
#     -6: 'dir/1',
#     -5: 'dir/2',
#     -4: 'dir/3',
#     -3: 'dir/4',
#     -2: 'light_level',
#     -1: 'is_sleeping',
# }

# map_feature_idxs = {
#     0: 'block/invalid',
#     1: 'block/out_of_bounds',
#     2: 'block/grass',
#     3: 'block/water',
#     4: 'block/stone',
#     5: 'block/tree',
#     6: 'block/wood',
#     7: 'block/path',
#     8: 'block/coal',
#     9: 'block/iron',
#     10: 'block/diamond',
#     11: 'block/crafting_table',
#     12: 'block/furnace',
#     13: 'block/sand',
#     14: 'block/lava',
#     15: 'block/plant',
#     16: 'block/ripe_plant',
#     17: 'mob/zombie',
#     18: 'mob/cow',
#     19: 'mob/skeleton',
#     20: 'mob/arrow'
# }

# # Replace negative indices with positive indices
# for idx in obs_feature_labels:
#     if idx < 0:
#         obs_feature_labels[idx + OBS_SIZE] = obs_feature_labels[idx]

# for h in range(MAP_HEIGHT):
#     for w in range(MAP_WIDTH):
#         for f in range(MAP_FEATURES):
#             map_feature_name = map_feature_idxs[f]
#             idx = (h * MAP_WIDTH * MAP_FEATURES) + (w * MAP_FEATURES) + f
#             obs_feature_labels[idx] = f'{map_feature_name}/h{h}-w{w}'

def crafter_obs_decomposed(state):
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Map
    padded_grid = jnp.pad(
        state.map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2

    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)
    map_view_one_hot = jax.nn.one_hot(map_view, num_classes=len(BlockType))

    # Mobs
    mob_map = jnp.zeros((*OBS_DIM, 4), dtype=jnp.uint8)  # 4 types of mobs

    def _add_mob_to_map(carry, mob_index):
        mob_map, mobs, mob_type_index = carry

        local_position = (
            mobs.position[mob_index]
            - state.player_position
            + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])
        ).all()
        on_screen *= mobs.mask[mob_index]

        mob_map = mob_map.at[local_position[0], local_position[1], mob_type_index].set(
            on_screen.astype(jnp.uint8)
        )

        return (mob_map, mobs, mob_type_index), None

    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.zombies, 0),
        jnp.arange(state.zombies.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map, (mob_map, state.cows, 1), jnp.arange(state.cows.mask.shape[0])
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.skeletons, 2),
        jnp.arange(state.skeletons.mask.shape[0]),
    )
    (mob_map, _, _), _ = jax.lax.scan(
        _add_mob_to_map,
        (mob_map, state.arrows, 3),
        jnp.arange(state.arrows.mask.shape[0]),
    )

    all_map = jnp.concatenate([map_view_one_hot, mob_map], axis=-1)

    # Inventory
    inventory = (
        jnp.array(
            [
                state.inventory.wood,
                state.inventory.stone,
                state.inventory.coal,
                state.inventory.iron,
                state.inventory.diamond,
                state.inventory.sapling,
                state.inventory.wood_pickaxe,
                state.inventory.stone_pickaxe,
                state.inventory.iron_pickaxe,
                state.inventory.wood_sword,
                state.inventory.stone_sword,
                state.inventory.iron_sword,
            ]
        ).astype(jnp.float16)
        / 10.0
    )

    intrinsics = (
        jnp.array(
            [
                state.player_health,
                state.player_food,
                state.player_drink,
                state.player_energy,
            ]
        ).astype(jnp.float16)
        / 10.0
    )

    direction = jax.nn.one_hot(state.player_direction - 1, num_classes=4)

    return (
        all_map,
        inventory,
        intrinsics,
        direction,
        jnp.array([state.light_level]),
        jnp.array([state.is_sleeping]),
    )

def embedding_crafter(next_obs, next_state):
    all_map, inventory, intrinsics, direction, light_level, is_sleeping = crafter_obs_decomposed(next_state)
    ic(all_map.shape, inventory.shape, intrinsics.shape, direction.shape, light_level.shape, is_sleeping.shape)
    raise NotImplementedError
