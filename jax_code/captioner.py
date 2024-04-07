from craftax_classic.envs.craftax_state import CraftaxState
from craftax_classic.constants import BlockType
import jax.numpy as jnp

def caption_state(state: CraftaxState):
    # TODO only count the blocks and mobs visible to the user (currently this is the whole map)

    blocks = jnp.unique(state.map)
    mobs = jnp.unique(state.mob_map)
    inventory = state.inventory

    # For each attr in inventory, store the attribute names for which the value is > 0 (assume inventory is a class)
    items = []
    for attr in dir(inventory):
        if attr.startswith('__'):
            continue
        if inventory[attr]:
            items.append(attr)

    block_names = []

    return blocks, mobs, inventory
