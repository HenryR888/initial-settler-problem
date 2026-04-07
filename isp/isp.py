from enum import IntEnum
import math
from typing import Any, Optional, Union
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from flax.struct import dataclass
import colorsys

from .multi_agent_env import MultiAgentEnv
from .spaces import Discrete, Dict, Tuple, Box


from .rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

# Define the Global Ground Truth State for the ISP Environment:

@dataclass
class State:
    agent_locs: jnp.ndarray # (n,3)...2d tensor, with rows representing agent index, and 3 columns - x,y, and direction agent is facing
    river_level: jnp.ndarray # scalar value...representing the global hidden resource (health) level of the river, R_t in [0,1]
    energy: jnp.ndarray     # (n,)...1d tensor (vector), representing private energy level, e_i in [0,1]
    reputations: jnp.ndarray # (n,)...1d tensor (vector), representing the public reoutation score for each agent, ro in [0,1] - updated this from ro in Reals, to make it simpler 
    cumulative_harvest: jnp.ndarray # (n,)...1d tensor (vector), representing cumulative harvest, which will be used in Greed Metric Q_{t,j} 
    cumulative_invest: jnp.ndarray # (n,)...1d tensor (vector), representing cumulative invest, which will also be used in Greed Metric Q_{t,j}
    num_steps_below_collapse: jnp.ndarray # number of consecutive steps for which R_t < K (collapse threshold)
    river_obs: jnp.ndarray # (n,)...1d tenstor vector, representing the per-agent noisy river estimate 
    grid: jnp.ndarray # (H,W)...that is the (height, width) of the spatial grid in which the agents operate
    inner_t: int # current timestep within episode
    outer_t: int # current episode index number

    # ! still need to add last_claims, which will be a vector which keeps track of last claim made by agents for the Lie metric...I will come back to this once adding communication action !


# Define the Action Set:

class Actions(IntEnum):
    turn_left = 0
    turn_right = 1
    left = 2
    right = 3
    up = 4
    down = 5
    stay = 6 # this will be equivalent to our NoOp action, in which the agent's energy level would change by clip(e_{t,i}+g,0,1)
    harvest = 7 # the agent harvests from the river (imagine it being water/fish), in which case the agent's energy increases by += beta_h, and the associated river damage (D_t,i) is -= gamma_h
    invest = 8 # the agent invests into maintaining/repairing river...energy reduces by -=beta_v, where beta_v > beta_h to ensure strategic non-triviality of pro-river-investment...moreover, associated river health contribution is gamma_v, where gamma_v > beta_v

    # ! Note: that Punish(j) is defined within the __init__ module further down, since we cannot parametrise Punish within this actions class object. We put it in __init__ module specifically, because that is where the number of agents first becomes known 

    # ! Note: we will come back to adding the communcation action set, as  I want to get the core dynamics of ISP working first before adding additional complexity


class Items(IntEnum):
    empty = 0
    wall = 1
    river = 2 # these shall be the river tiles for our environment...the agents will be able to harvest/invest when standing on or adjacent to these tiles.


# 0th index is for row changes; 1st index is for column changes; 2nd index is for direction changes
ROTATIONS = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # left
        [0, 0, 0],  # right
        [0, 0, 0],  # up
        [0, 0, 0],  # down
        [0, 0, 0],  # stay
        [0, 0, 0],  # harvest (does not change direction that the agent is facing)
        [0, 0, 0],  # invest (does not change direction that the agent is facing)
        
        # ! Note: We will add the Punish(j) row dynamically within the __init__ module 
    ],
    dtype=jnp.int8,
)

STEP = jnp.array(
    [
        [1, 0, 0],  # up
        [0, 1, 0],  # right
        [-1, 0, 0],  # down
        [0, -1, 0],  # left
    ],
    dtype=jnp.int8,
)

STEP_MOVE = jnp.array(
    [
        [0, 0, 0], # turn left
        [0, 0, 0], # turn right
        [0, 1, 0], # left
        [0, -1, 0], # right  
        [1, 0, 0], # up  
        [-1, 0, 0], # down  
        [0, 0, 0], # stay
        [0, 0, 0], # harvest
        [0, 0, 0], # invest

         # ! Note: We will add the Punish(j) row dynamically within the __init__ module 
    ],
    dtype=jnp.int8,
)

char_to_int = {
    ' ': 0, # empty tile
    'W': 1, # wall
    'R': 2, # river tile
    'P': 3, # spawn point for agents, which is only used when we reset environement
}

def ascii_map_to_matrix(map_ASCII, char_to_int):
    """
    Convert ASCII map to a JAX numpy matrix using the given character mapping.
    
    Args:
    map_ASCII (list): List of strings representing the ASCII map
    char_to_int (dict): Dictionary mapping characters to integer values
    
    Returns:
    jax.numpy.ndarray: 2D matrix representation of the ASCII map
    """
    # Determine matrix dimensions
    height = len(map_ASCII)
    width = max(len(row) for row in map_ASCII)
    
    # Create matrix filled with zeros
    matrix = jnp.zeros((height, width), dtype=jnp.int32)
    
    # Fill matrix with mapped values
    for i, row in enumerate(map_ASCII):
        for j, char in enumerate(row):
            matrix = matrix.at[i, j].set(char_to_int.get(char, 0))
    
    return matrix

# create colours for agent arrows in visualisation
def generate_agent_colors(num_agents): 
    colors = []
    for i in range(num_agents):
        hue = i / num_agents
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)  # Saturation and Value set to 0.8
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors

GREEN_COLOUR = (44.0, 160.0, 44.0)
RED_COLOUR = (214.0, 39.0, 40.0)
###################################################

class Clean_up(MultiAgentEnv):

    # used for caching
    tile_cache = {}

    def __init__(
        self,
        num_inner_steps=1000, # length of an episode
        num_outer_steps=1, # number of episodes ran (initialising here)
        num_agents=7,
        reward_type="shared",  # "shared", "individual", or "saturating"
        inequity_aversion=False,
        inequity_aversion_target_agents=None,
        inequity_aversion_alpha=5,
        inequity_aversion_beta=0.05,
        svo=False,
        svo_target_agents=None,
        svo_w=0.5,
        svo_ideal_angle_degrees=45,
        enable_smooth_rewards=False,
        maxAppleGrowthRate=0.05, 
        thresholdDepletion=0.4,  # 0.4 - note that this directly affects apple growth. As soon as the apple regrowth rate will start collapsing
        thresholdRestoration=0.0,
        dirtSpawnProbability=0.5,
        delayStartOfDirtSpawning=50, # 50
        jit=True,
        obs_size=11,
        cnn=True,
        agent_ids=False,

        # This is a 15x9 gridworld which we have proposed, on the basis that we will use 3 agents to start out. There are walls on the left and right and bottom of the environment to prevent the agents walking out. 
        # the reason for the wall between the river and the agents living area is to prevent an agent just being able stand next to a river for infinite time and continue harvesting/investing...they actually need to make a choice to leave the living area and enter the river to harvest/invest
        map_ASCII = [
                'WRRRRRRRRRRRRRW',
                'WRRRRRRRRRRRRRW',
                'WRRRRRRRRRRRRRW',
                'WWWWWWWWWWWWWWW',
                '   P    P    P ',
                '  P    P    P  ',
                '   P    P    P ',
                'WWWWWWWWWWWWWWW',
            ]
    ):

        super().__init__(num_agents=num_agents)

        self.maxAppleGrowthRate = maxAppleGrowthRate
        self.thresholdDepletion = thresholdDepletion
        self.thresholdRestoration = thresholdRestoration
        self.dirtSpawnProbability = dirtSpawnProbability
        self.delayStartOfDirtSpawning = delayStartOfDirtSpawning
        self.reward_type = reward_type
        self.inequity_aversion = inequity_aversion
        self.inequity_aversion_target_agents = inequity_aversion_target_agents
        self.inequity_aversion_alpha = inequity_aversion_alpha
        self.inequity_aversion_beta = inequity_aversion_beta
        self.svo = svo
        self.svo_target_agents = svo_target_agents
        self.svo_w = svo_w
        self.svo_ideal_angle_degrees = svo_ideal_angle_degrees
        self.smooth_rewards = enable_smooth_rewards
        self.cnn = cnn
        self.agent_ids = agent_ids
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps

        self.agents = list(range(num_agents))#, dtype=jnp.int16)
        self._agents = jnp.array(self.agents, dtype=jnp.int16) + len(Items)

        self.PLAYER_COLOURS = generate_agent_colors(num_agents)
        self.GRID_SIZE_ROW = len(map_ASCII)
        self.GRID_SIZE_COL = len(map_ASCII[0])
        self.OBS_SIZE = obs_size
        self.PADDING = self.OBS_SIZE - 1

        GRID = jnp.zeros(
            (self.GRID_SIZE_ROW + 2 * self.PADDING, self.GRID_SIZE_COL + 2 * self.PADDING),
            dtype=jnp.int16,
        )

        # First layer of padding is Wall
        GRID = GRID.at[self.PADDING - 1, :].set(5)
        GRID = GRID.at[self.GRID_SIZE_ROW + self.PADDING, :].set(5)
        GRID = GRID.at[:, self.PADDING - 1].set(5)
        self.GRID = GRID.at[:, self.GRID_SIZE_COL + self.PADDING].set(5)

        def find_positions(grid_array, letter):
            a_positions = jnp.array(jnp.where(grid_array == letter)).T
            return a_positions

        nums_map = ascii_map_to_matrix(map_ASCII, char_to_int)
        self.POTENTIAL_APPLE = find_positions(nums_map, char_to_int['B'])

        self.SPAWNS_PLAYER_IN = find_positions(nums_map, char_to_int['Q'])
        self.SPAWNS_PLAYERS = find_positions(nums_map, char_to_int['P'])
        self.SPAWNS_WALL = find_positions(nums_map, char_to_int['W'])
        self.RIVER = find_positions(nums_map, char_to_int['S'])
        self.POTENTIAL_DIRT = find_positions(nums_map, char_to_int['H'])
        self.DIRT = find_positions(nums_map, char_to_int['F'])

        def check_collision(
                new_agent_locs: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function to check agent collisions.
            
            Args:
                - new_agent_locs: jnp.ndarray, the agent locations at the 
                current time step.
                
            Returns:
                - jnp.ndarray matrix of bool of agents in collision.
            '''
            matcher = jax.vmap(
                lambda x,y: jnp.all(x[:2] == y[:2]),
                in_axes=(0, None)
            )

            collisions = jax.vmap(
                matcher,
                in_axes=(None, 0)
            )(new_agent_locs, new_agent_locs)

            return collisions
        
        def fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Function defining multi-collision logic.
            If agents move into tile, and there is already and agent in that tile,
            then agent who was there has right of way, and the other agents get reset to where they were.
            Case 2: if all agents moved to a specfic tile, then one will win at random, and the others who lose out
            on the tile will be reset to their previous tile.

            Args:
                - key: jax key for randomisation
                - collided_moved: jnp.ndarray, the agents which moved in the
                last time step and caused collisions.
                - collision_matrix: jnp.ndarray, the agents currently in
                collisions
                - agent_locs: jnp.ndarray, the agent locations at the previous
                time step.
                - new_agent_locs: jnp.ndarray, the agent locations at the
                current time step.

            Returns:
                - jnp.ndarray of the final positions after collisions are
                managed.
            """
            def scan_fn(
                    state,
                    idx
            ):
                key, collided_moved, collision_matrix, agent_locs, new_agent_locs = state

                return jax.lax.cond(
                    collided_moved[idx] > 0,
                    lambda: _fix_collisions(
                        key,
                        collided_moved,
                        collision_matrix,
                        agent_locs,
                        new_agent_locs
                    ),
                    lambda: (state, new_agent_locs)
                )

            _, ys = jax.lax.scan(
                scan_fn,
                (key, collided_moved, collision_matrix, agent_locs, new_agent_locs),
                jnp.arange(self.num_agents)
            )

            final_locs = ys[-1]

            return final_locs

        def _fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ):
            def select_random_true_index(key, array):
                # Calculate the cumulative sum of True values
                cumsum_array = jnp.cumsum(array)

                # Count the number of True values
                true_count = cumsum_array[-1]

                # Generate a random index in the range of the number of True
                # values
                rand_index = jax.random.randint(
                    key,
                    (),
                    0,
                    true_count
                )

                # Find the position of the random index within the cumulative
                # sum
                chosen_index = jnp.argmax(cumsum_array > rand_index)

                return chosen_index
            # Pick one from all who collided & moved
            colliders_idx = jnp.argmax(collided_moved)

            collisions = collision_matrix[colliders_idx]

            # Check whether any of collision participants didn't move
            collision_subjects = jnp.where(
                collisions,
                collided_moved,
                collisions
            )
            collision_mask = collisions == collision_subjects
            stayed = jnp.all(collision_mask)
            stayed_mask = jnp.logical_and(~stayed, ~collision_mask)
            stayed_idx = jnp.where(
                jnp.max(stayed_mask) > 0,
                jnp.argmax(stayed_mask),
                0
            )

            # Prepare random agent selection
            k1, k2 = jax.random.split(key, 2)
            rand_idx = select_random_true_index(k1, collisions)
            collisions_rand = collisions.at[rand_idx].set(False)
            new_locs_rand = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_rand,
                agent_locs,
                new_agent_locs
            )

            collisions_stayed = jax.lax.select(
                jnp.max(stayed_mask) > 0,
                collisions.at[stayed_idx].set(False),
                collisions_rand
            )
            new_locs_stayed = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_stayed,
                agent_locs,
                new_agent_locs
            )

            # Choose between the two scenarios - revert positions if
            # non-mover exists, otherwise choose random agent if all moved
            new_agent_locs = jnp.where(
                stayed,
                new_locs_rand,
                new_locs_stayed
            )

            # Update move bools to reflect the post-collision positions
            collided_moved = jnp.clip(collided_moved - collisions, 0, 1)
            collision_matrix = collision_matrix.at[colliders_idx].set(
                [False] * collisions.shape[0]
            )
            return ((k2, collided_moved, collision_matrix, agent_locs, new_agent_locs), new_agent_locs)
        
        def combine_channels(
                grid: jnp.ndarray,
                agent: int,
                angles: jnp.ndarray,
                agent_pickups: jnp.ndarray,
                state: State,
            ):
            """
            This function converts a one-hot encoded vector into another one-hot
            encoded vector that is much more expressive and suitable for learning behaviour.
            """
            def move_and_collapse(
                    x: jnp.ndarray,
                    angle: jnp.ndarray,
                ) -> jnp.ndarray:

                # get agent's one-hot
                agent_element = jnp.array([jnp.int8(x[agent])])

                # mask to check if any other agent exists there
                mask = x[len(Items)-1:] > 0

                # does an agent exist which is not the subject?
                other_agent = jnp.int8(
                    jnp.logical_and(
                        jnp.any(mask),
                        jnp.logical_not(
                            agent_element
                        )
                    )
                )

                # what is the class of the item in cell
                item_idx = jnp.where(
                    x,
                    size=1
                )[0]

                # check if agent is frozen and can observe inventories
                show_inv_bool = jnp.logical_and(
                        state.freeze[
                            agent - len(Items)
                        ].max(axis=-1) > 0,
                        item_idx >= len(Items)
                )

                show_inv_idxs = jnp.nonzero(
                    state.freeze[agent - len(Items)],
                    size=12,
                    fill_value=-1
                )[0]

                inv_to_show = jnp.where(
                    jnp.logical_or(
                        jnp.logical_and(
                            show_inv_bool,
                            jnp.any((item_idx - len(Items)) == show_inv_idxs),
                        ),
                        agent_element
                    ),
                    state.agent_invs[item_idx - len(Items)],
                    jnp.array([0, 0], dtype=jnp.int8)
                )[0]

                # check if agent is not the subject & is frozen & therefore
                # not possible to interact with
                frozen = jnp.where(
                    other_agent,
                    state.freeze[
                        item_idx-len(Items)
                    ].max(axis=-1) > 0,
                    0
                )

                # get pickup/inv info
                pick_up_idx = jnp.where(
                    jnp.any(mask),
                    jnp.nonzero(mask, size=1)[0],
                    jnp.int8(-1)
                )
                picked_up = jnp.where(
                    pick_up_idx > -1,
                    agent_pickups[pick_up_idx],
                    jnp.int8(0)
                )

                # build extension
                extension = jnp.concatenate(
                    [
                        agent_element,
                        other_agent,
                        angle,
                        picked_up,
                        inv_to_show,
                        frozen
                    ],
                    axis=-1
                )

                # build final feature vector
                final_vec = jnp.concatenate(
                    [x[:len(Items)-1], extension],
                    axis=-1
                )

                return final_vec

            new_grid = jax.vmap(
                jax.vmap(
                    move_and_collapse
                )
            )(grid, angles)
            return new_grid
        
        def check_relative_orientation(
                agent: int,
                agent_locs: jnp.ndarray,
                grid: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Compute relative orientations of other agents visible to the current agent.
            Returns -1 where no other agent is present.
            This is important because this will allow agents to avoid being blind to intentions of other agents.
            '''
            idx = agent - len(Items)
            curr_agent_dir = agent_locs[idx, 2]

            first_agent_id = len(Items)
            last_agent_id = len(Items) + self.num_agents - 1

            mask = (grid >= first_agent_id) & (grid <= last_agent_id) & (grid != agent)

            # Safe gather: map cell id -> agent index [0, num_agents-1]
            agent_index = jnp.clip(grid - len(Items), 0, self.num_agents - 1).astype(jnp.int32)
            cell_direction = agent_locs[agent_index, 2]

            rel_dir = (cell_direction - curr_agent_dir) % 4
            angle = jnp.where(mask, rel_dir, -1)

            return angle
        
        def rotate_grid(agent_loc: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
            '''
            Rotates agent's observation grid by 0/90/180/270 degrees based on direction.
            Uses a data-dependent branch to avoid computing all rotations. 
            This aligns the world to that specific agent's facing-direction (egocentric), so that the CNN policy
            training becomes dramatically easier.
            '''
            def rot0(g):
                return g
            def rot1(g):
                return jnp.rot90(g, k=1, axes=(0, 1))
            def rot2(g):
                return jnp.rot90(g, k=2, axes=(0, 1))
            def rot3(g):
                return jnp.rot90(g, k=3, axes=(0, 1))
            return jax.lax.switch(jnp.asarray(agent_loc[2], jnp.int32), [rot0, rot1, rot2, rot3], grid)

        def _get_obs_point(agent_loc: jnp.ndarray) -> jnp.ndarray:
            '''
            Obtain the position of top-left corner of obs map using
            agent's current location & orientation.

            Args: 
                - agent_loc: jnp.ndarray, agent x, y, direction.
            Returns:
                - x, y: ints of top-left corner of agent's obs map.
            '''
            
            x, y, direction = agent_loc

            x, y = x + self.PADDING, y + self.PADDING

            x = x - (self.OBS_SIZE // 2)
            y = y - (self.OBS_SIZE // 2)


            x = jnp.where(direction == 0, x + (self.OBS_SIZE//2)-1, x)
            y = jnp.where(direction == 0, y, y)

            x = jnp.where(direction == 1, x, x)
            y = jnp.where(direction == 1, y + (self.OBS_SIZE//2)-1, y)


            x = jnp.where(direction == 2, x - (self.OBS_SIZE//2)+1, x)
            y = jnp.where(direction == 2, y, y)


            x = jnp.where(direction == 3, x, x)
            y = jnp.where(direction == 3, y - (self.OBS_SIZE//2)+1, y)
            return x, y

        def _get_obs(state: State) -> jnp.ndarray:
            '''
            Obtain the agent's observation of the grid.

            Args: 
                - state: State object containing env state.
            Returns:
                - jnp.ndarray of grid observation.
            '''
            # create state
            grid = jnp.pad(
                state.grid,
                ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)),
                constant_values=Items.wall,
            )

            # obtain all agent obs-points
            agent_start_idxs = jax.vmap(_get_obs_point)(state.agent_locs)

            dynamic_slice = partial(
                jax.lax.dynamic_slice,
                operand=grid,
                slice_sizes=(self.OBS_SIZE, self.OBS_SIZE)
            )

            # obtain agent obs grids
            grids = jax.vmap(dynamic_slice)(start_indices=agent_start_idxs)

            # rotate agent obs grids
            grids = jax.vmap(rotate_grid)(state.agent_locs, grids)

            angles = jax.vmap(
                check_relative_orientation,
                in_axes=(0, None, 0)
            )(
                self._agents,
                state.agent_locs,
                grids
            )

            angles = jax.nn.one_hot(angles, 4)

            # one-hot (drop first channel as its empty blocks)...this one-hot encoding of the grid is what gets input into our CNN in our ippo.py file
            grids = jax.nn.one_hot(
                grids - 1,
                num_agents + len(Items) - 1, # will be collapsed into a
                dtype=jnp.int8 # [Items, self, other, extra features] representation
            )

            # check agents that can interact
            inventory_sum = jnp.sum(state.agent_invs, axis=-1)
            agent_pickups = jnp.where(
                inventory_sum > INTERACT_THRESHOLD,
                True,
                False
            )

            # make index len(Item) always the current agent
            # and sum all others into an "other" agent
            grids = jax.vmap(
                combine_channels,
                in_axes=(0, 0, 0, None, None)
            )(
                grids,
                self._agents,
                angles,
                agent_pickups,
                state
            )

            # Add coefficient channel showing relative apple collection
            def add_coef_channel(grid, agent_idx):
                # Calculate coefficient for this agent
                agent_apple_counts = state.cumulative_apples_collected
                max_apples = jnp.max(agent_apple_counts)

                coef = jax.lax.cond(
                    max_apples == 0,
                    lambda _: jnp.array(1.0, dtype=jnp.bfloat16),
                    lambda _: jnp.asarray(1.0, dtype=jnp.bfloat16) - jnp.asarray(agent_apple_counts[agent_idx], dtype=jnp.bfloat16) / jnp.asarray(max_apples, dtype=jnp.bfloat16) + jnp.asarray(0.2, dtype=jnp.bfloat16),
                    operand=max_apples,
                )
                
                # Create channel filled with coefficient value
                coef_channel = jnp.full(
                    (self.OBS_SIZE, self.OBS_SIZE, 1),
                    coef,
                    dtype=jnp.int8,
                )
                
                # Concatenate with existing observation
                return jnp.concatenate([grid, coef_channel], axis=-1)

            # Apply coefficient channel to all agent observations
            grids = jax.vmap(add_coef_channel, in_axes=(0, 0))(grids, jnp.arange(num_agents))

            # Add agent ID channels if enabled
            if self.agent_ids:
                # Create agent ID channels for each agent
                def add_agent_id_channels(grid, agent_idx):
                    # Create num_agents channels, all zeros
                    agent_id_channels = jnp.zeros((self.OBS_SIZE, self.OBS_SIZE, num_agents), dtype=jnp.int8)
                    # Set the channel corresponding to this agent's ID to all ones
                    agent_id_channels = agent_id_channels.at[:, :, agent_idx].set(1)
                    # Concatenate with existing observation
                    return jnp.concatenate([grid, agent_id_channels], axis=-1)

                # Apply agent ID channels to all agent observations
                grids = jax.vmap(add_agent_id_channels, in_axes=(0, 0))(grids, self._agents)

            return grids


        def _interact_fire_zapping(
            key: jnp.ndarray, state: State, actions: jnp.ndarray
        ):
            '''
            Main interaction logic entry point.
            This just returns which agents, if any were zapped.

            Args:
                - key: jax key for randomisation.
                - state: State env state object.
                - actions: jnp.ndarray of actions taken by agents.
            Returns:
                - (jnp.ndarray, State, jnp.ndarray) - Tuple where index 0 is
                the array of rewards obtained, index 2 is the new env State,
                and index 3 is the new freeze penalty matrix.
            '''
            # if interact
            zaps = actions == Actions.zap_forward

            interact_idx = jnp.int16(Items.interact)

            # Always clear last-step beams so they are single-frame
            state = state.replace(grid=jnp.where(
                state.grid == interact_idx, jnp.int16(Items.empty), state.grid
            ))
            state = state.replace(grid=jnp.where(
                state.grid == Items.clean_beam, jnp.int16(Items.empty), state.grid
            ))
            any_zap = jnp.any(zaps)

            # calculate pickups
            # agent_pickups = state.agent_invs.sum(axis=-1) > -100
            def zap_branch(st: State):
                one_step_targets = jax.vmap(
                    lambda p: p + STEP[p[2]]
                )(st.agent_locs)

                two_step_targets = jax.vmap(
                    lambda p: p + 2 * STEP[p[2]]
                )(st.agent_locs)

                target_right = jax.vmap(
                    lambda p: p + STEP[p[2]] + STEP[(p[2] + 1) % 4]
                )(st.agent_locs)

                target_left = jax.vmap(
                    lambda p: p + STEP[p[2]] + STEP[(p[2] - 1) % 4]
                )(st.agent_locs)

                # Vectorized OOB checks on XY only
                max_x = self.GRID_SIZE_ROW - 1
                max_y = self.GRID_SIZE_COL - 1
                right_oob_check = (
                    (target_right[:, 0] > max_x) | (target_right[:, 0] < 0) |
                    (target_right[:, 1] > max_y) | (target_right[:, 1] < 0)
                )
                left_oob_check = (
                    (target_left[:, 0] > max_x) | (target_left[:, 0] < 0) |
                    (target_left[:, 1] > max_y) | (target_left[:, 1] < 0)
                )

                target_right = jnp.where(
                    right_oob_check[:, None],
                    one_step_targets,
                    target_right
                )
                target_left = jnp.where(
                    left_oob_check[:, None],
                    one_step_targets,
                    target_left
                )

                all_zaped_locs = jnp.concatenate((one_step_targets, two_step_targets, target_right, target_left), 0)
                zaps_4_locs = jnp.concatenate((zaps, zaps, zaps, zaps), 0)

                def zaped_gird(a, z):
                    return jnp.where(z, st.grid[a[0], a[1]], -1)

                all_zaped_gird = jax.vmap(zaped_gird)(all_zaped_locs, zaps_4_locs)

                def check_reborn_player(a):
                    return jnp.any(all_zaped_gird == a)

                reborn_players = jax.vmap(check_reborn_player)(self._agents)

                aux_grid = st.grid

                o_items = jnp.where(
                    st.grid[
                        one_step_targets[:, 0],
                        one_step_targets[:, 1]
                    ],
                    st.grid[
                        one_step_targets[:, 0],
                        one_step_targets[:, 1]
                    ],
                    interact_idx
                )
                t_items = jnp.where(
                    st.grid[
                        two_step_targets[:, 0],
                        two_step_targets[:, 1]
                    ],
                    st.grid[
                        two_step_targets[:, 0],
                        two_step_targets[:, 1]
                    ],
                    interact_idx
                )
                r_items = jnp.where(
                    st.grid[
                        target_right[:, 0],
                        target_right[:, 1]
                    ],
                    st.grid[
                        target_right[:, 0],
                        target_right[:, 1]
                    ],
                    interact_idx
                )
                l_items = jnp.where(
                    st.grid[
                        target_left[:, 0],
                        target_left[:, 1]
                    ],
                    st.grid[
                        target_left[:, 0],
                        target_left[:, 1]
                    ],
                    interact_idx
                )

                qualified_to_zap = zaps.squeeze()

                def update_grid(a_i, t, i, grid):
                    return grid.at[t[:, 0], t[:, 1]].set(
                        jax.vmap(jnp.where)(
                            a_i,
                            i,
                            aux_grid[t[:, 0], t[:, 1]]
                        )
                    )

                aux_grid2 = update_grid(qualified_to_zap, one_step_targets, o_items, aux_grid)
                aux_grid2 = update_grid(qualified_to_zap, two_step_targets, t_items, aux_grid2)
                aux_grid2 = update_grid(qualified_to_zap, target_right, r_items, aux_grid2)
                aux_grid2 = update_grid(qualified_to_zap, target_left, l_items, aux_grid2)

                st2 = st.replace(grid=aux_grid2)
                return reborn_players, st2

            def no_zap_branch(st: State):
                return jnp.zeros((self.num_agents,), dtype=jnp.bool_), st

            reborn_players, state = jax.lax.cond(any_zap, zap_branch, no_zap_branch, state)
            return reborn_players, state
        
        def _interact_fire_cleaning(
            key: jnp.ndarray, state: State, actions: jnp.ndarray
        ):
            '''
            Main interaction logic entry point. 
            This function defines how the cleaning function works: 
            When an agent chooses to clean, it can affect up to four tiles in
            front of where the agent chose to clean. Once cleaned, there is a 
            clean_beam marker so that other agents can observe who cleaned the grid. 
            The tile that was cleaned, then changes to a 'Potential_dirt grid'

            Args:
                - key: jax key for randomisation.
                - state: State env state object.
                - actions: jnp.ndarray of actions taken by agents.
            Returns:
                - (jnp.ndarray, State, jnp.ndarray) - Tuple where index 0 is
                the array of rewards obtained, index 2 is the new env State,
                and index 3 is the new freeze penalty matrix.
            '''
            # if interact
            zaps = actions == Actions.zap_clean

            interact_idx = jnp.int16(Items.clean_beam)

            any_zap = jnp.any(zaps)
            # Always clear last-step cleaning beams
            state = state.replace(grid=jnp.where(
                state.grid == interact_idx, jnp.int16(Items.empty), state.grid
            ))

            def clean_branch(st: State):
                # Remove old beams of same type
                g = jnp.where(st.grid == interact_idx, jnp.int16(Items.empty), st.grid)
                st = st.replace(grid=g)

                one_step_targets = jax.vmap(
                    lambda p: p + STEP[p[2]]
                )(st.agent_locs)
                two_step_targets = jax.vmap(
                    lambda p: p + 2 * STEP[p[2]]
                )(st.agent_locs)
                target_right = jax.vmap(
                    lambda p: p + STEP[p[2]] + STEP[(p[2] + 1) % 4]
                )(st.agent_locs)
                target_left = jax.vmap(
                    lambda p: p + STEP[p[2]] + STEP[(p[2] - 1) % 4]
                )(st.agent_locs)

                max_x = self.GRID_SIZE_ROW - 1
                max_y = self.GRID_SIZE_COL - 1
                right_oob_check = (
                    (target_right[:, 0] > max_x) | (target_right[:, 0] < 0) |
                    (target_right[:, 1] > max_y) | (target_right[:, 1] < 0)
                )
                left_oob_check = (
                    (target_left[:, 0] > max_x) | (target_left[:, 0] < 0) |
                    (target_left[:, 1] > max_y) | (target_left[:, 1] < 0)
                )
                target_right = jnp.where(right_oob_check[:, None], one_step_targets, target_right)
                target_left = jnp.where(left_oob_check[:, None], one_step_targets, target_left)

                all_zaped_locs = jnp.concatenate((one_step_targets, two_step_targets, target_right, target_left), 0)
                zaps_4_locs_judge = jnp.concatenate((zaps, zaps, zaps, zaps), 0)

                potential_dirt_all_zap = jnp.repeat(jnp.array(Items.potential_dirt), len(all_zaped_locs))

                def clean_gird(a, judge):
                    return st.grid.at[a[:, 0], a[:, 1]].set(
                        jax.vmap(jnp.where)(
                            ((judge == True) & (st.grid[a[:, 0], a[:, 1]] == Items.dirt)),
                            potential_dirt_all_zap,
                            st.grid[a[:, 0], a[:, 1]]
                        )
                    )

                grid_clean = clean_gird(all_zaped_locs, zaps_4_locs_judge.squeeze())
                st = st.replace(grid=grid_clean)

                def renew_dirt_label(locs, labels):
                    v = grid_clean[locs[0], locs[1]]
                    return jnp.where((v == Items.dirt) | (v == Items.potential_dirt), v, labels)

                renew_label = jax.vmap(renew_dirt_label)(st.potential_dirt_and_dirt_locs, st.potential_dirt_and_dirt_label)
                st = st.replace(potential_dirt_and_dirt_label=renew_label)

                aux_grid = st.grid
                o_items = jnp.where(
                    st.grid[
                        one_step_targets[:, 0],
                        one_step_targets[:, 1]
                    ],
                    st.grid[
                        one_step_targets[:, 0],
                        one_step_targets[:, 1]
                    ],
                    interact_idx
                )
                t_items = jnp.where(
                    st.grid[
                        two_step_targets[:, 0],
                        two_step_targets[:, 1]
                    ],
                    st.grid[
                        two_step_targets[:, 0],
                        two_step_targets[:, 1]
                    ],
                    interact_idx
                )
                r_items = jnp.where(
                    st.grid[
                        target_right[:, 0],
                        target_right[:, 1]
                    ],
                    st.grid[
                        target_right[:, 0],
                        target_right[:, 1]
                    ],
                    interact_idx
                )
                l_items = jnp.where(
                    st.grid[
                        target_left[:, 0],
                        target_left[:, 1]
                    ],
                    st.grid[
                        target_left[:, 0],
                        target_left[:, 1]
                    ],
                    interact_idx
                )

                qualified_to_zap = zaps.squeeze()

                def update_grid(a_i, t, i, grid):
                    return grid.at[t[:, 0], t[:, 1]].set(
                        jax.vmap(jnp.where)(
                            a_i,
                            i,
                            aux_grid[t[:, 0], t[:, 1]]
                        )
                    )

                aux_grid2 = update_grid(qualified_to_zap, one_step_targets, o_items, aux_grid)
                aux_grid2 = update_grid(qualified_to_zap, two_step_targets, t_items, aux_grid2)
                aux_grid2 = update_grid(qualified_to_zap, target_right, r_items, aux_grid2)
                aux_grid2 = update_grid(qualified_to_zap, target_left, l_items, aux_grid2)

                st = st.replace(grid=aux_grid2)
                return st

            state = jax.lax.cond(any_zap, clean_branch, lambda s: s, state)
            return state


        def _step(
            key: chex.PRNGKey,
            state: State,
            actions: jnp.ndarray
        ):
            """Step the environment."""

            # Split PRNG upfront for determinism and to avoid correlation between these different elements of randonmness.
            # That is to say, if we used one key for everything, then apple respawning randomness would be correlated with dirt spawning randomness.
            key, k_apple, k_dirt_noise, k_dirt_p, k_collision, k_respawn_perm, k_respawn_dir = jax.random.split(key, 7)

            # regrowth of apple
            grid_apple = state.grid
            dirtCount = jnp.sum(state.potential_dirt_and_dirt_label == Items.dirt)
            denom = jnp.asarray(len(state.potential_dirt_and_dirt_locs) + len(self.RIVER), dtype=jnp.bfloat16) 
            dirtFraction = jnp.asarray(dirtCount, dtype=jnp.bfloat16) / denom
            depletion = jnp.asarray(self.thresholdDepletion, dtype=jnp.bfloat16)
            restoration = jnp.asarray(self.thresholdRestoration, dtype=jnp.bfloat16)
            interpolation = (dirtFraction - depletion) / (restoration - depletion)

            interpolation = jnp.clip(
                interpolation,
                jnp.finfo(jnp.bfloat16).min,
                jnp.asarray(1.0, dtype=jnp.bfloat16),
            )
            probability = jnp.asarray(self.maxAppleGrowthRate, dtype=jnp.bfloat16) * interpolation
            def regrow_apple(apple_locs, p):
                '''
                Here, we are placing apples in the grid where there are empty spaces, and if our parameters p is less than prob.
                Note that when our AppleGrowthRate is low, then we simulate an environment which is quite sparse in regenerating apples.
                We shall be using a logistic growth rate for our ISP, but it would be interesting to vary the growth rate and see how the dynamics of the env play out.
                '''
                new_apple = jnp.where((((grid_apple[apple_locs[0], apple_locs[1]] == Items.empty) & (p < probability)) 
                                       | ((grid_apple[apple_locs[0], apple_locs[1]] == Items.apple))),  
                                      Items.apple, Items.empty)
                return new_apple
            prob = jax.random.uniform(k_apple, shape=(len(self.POTENTIAL_APPLE),), dtype=jnp.bfloat16)
            new_apple = jax.vmap(regrow_apple)(self.POTENTIAL_APPLE, prob) # this is the JAX way of running a loop. Effectively,
            # we are creating a vectorised map which spawns an apple in all the potential apple places, using the logic applied to the prob param all in parallel. 

            #Since JAX arrays are immutable, we need to create a new array with the apples added
            new_apple_grid = grid_apple.at[self.POTENTIAL_APPLE[:, 0], self.POTENTIAL_APPLE[:, 1]].set(new_apple)
            state = state.replace(grid=new_apple_grid)

            # DirtSpawning update the grid and potential_dirt_and_dirt_label
            grid_dirt = state.grid

            noise = jax.random.uniform(
                k_dirt_noise,
                shape=(len(state.potential_dirt_and_dirt_label),),
                dtype=jnp.bfloat16,
            ) * jnp.asarray(1e-4, dtype=jnp.bfloat16)
            label_with_noise = state.potential_dirt_and_dirt_label + noise

            label_with_noise_rank = jnp.sort(label_with_noise)
            unstable_indices = jnp.argsort(label_with_noise)

            unstable_sorted_locs = state.potential_dirt_and_dirt_locs[unstable_indices]
            
            p = jax.random.uniform(k_dirt_p, shape=(1,), dtype=jnp.bfloat16) 
            one_piece_dirt = jnp.where(((grid_dirt[unstable_sorted_locs[0, 0], unstable_sorted_locs[0, 1]] == Items.potential_dirt) 
                                       & (p < jnp.asarray(self.dirtSpawnProbability, dtype=jnp.bfloat16)) & (state.inner_t>self.delayStartOfDirtSpawning)),  
                        Items.dirt, label_with_noise_rank[0])

            label_with_noise_rank_new = label_with_noise_rank.at[0].set(one_piece_dirt[0]) 

            label_rank_new = jnp.round(label_with_noise_rank_new).astype(jnp.int16)

            state = state.replace(potential_dirt_and_dirt_label=label_rank_new)
            state = state.replace(potential_dirt_and_dirt_locs=unstable_sorted_locs)
            actions = jnp.array(actions)

            new_grid = state.grid.at[
                state.agent_locs[:, 0],
                state.agent_locs[:, 1]
            ].set(
                jnp.int16(Items.empty)
            )
            # first apply dirt and get that new array
            new_grid = new_grid.at[state.potential_dirt_and_dirt_locs[:, 0], state.potential_dirt_and_dirt_locs[:, 1]].set(state.potential_dirt_and_dirt_label)
            # then apply river on top
            new_grid = new_grid.at[self.RIVER[:, 0], self.RIVER[:, 1]].set(Items.river)

            # this is the way in which we shall place the agents on the grid: 
            x, y = state.reborn_locs[:, 0], state.reborn_locs[:, 1]
            new_grid = new_grid.at[x, y].set(self._agents)
            state = state.replace(grid=new_grid)
            state = state.replace(agent_locs=state.reborn_locs)

            all_new_locs = jax.vmap(lambda p, a: jnp.int16(p + ROTATIONS[a]) % jnp.array([self.GRID_SIZE_ROW + 1, self.GRID_SIZE_COL + 1, 4], dtype=jnp.int16))(p=state.agent_locs, a=actions).squeeze()

            agent_move = (actions == Actions.up) | (actions == Actions.down) | (actions == Actions.right) | (actions == Actions.left)
            all_new_locs = jax.vmap(lambda m, n, p: jnp.where(m, n + STEP_MOVE[p], n))(m=agent_move, n=all_new_locs, p=actions)
            
            all_new_locs = jax.vmap(
                jnp.clip,
                in_axes=(0, None, None)
            )(
                all_new_locs,
                jnp.array([0, 0, 0], dtype=jnp.int16),
                jnp.array(
                    [self.GRID_SIZE_ROW - 1, self.GRID_SIZE_COL - 1, 3],
                    dtype=jnp.int16
                ),
            ).squeeze()

            # if you bounced back to your original space,
            # change your move to stay (for collision logic)
            agents_move = jax.vmap(lambda n, p: jnp.any(n[:2] != p[:2]))(n=all_new_locs, p=state.agent_locs)

            # generate bool mask for agents colliding
            collision_matrix = check_collision(all_new_locs)

            # sum & subtract "self-collisions"
            collisions = jnp.sum(
                collision_matrix,
                axis=-1,
                dtype=jnp.int8
            ) - 1
            collisions = jnp.minimum(collisions, 1)

            # identify which of those agents made wrong moves
            collided_moved = jnp.maximum(
                collisions - ~agents_move,
                0
            )

            # fix collisions at the correct indices
            new_locs = jax.lax.cond(
                jnp.max(collided_moved) > 0,
                lambda: fix_collisions(
                    k_collision,
                    collided_moved,
                    collision_matrix,
                    state.agent_locs,
                    all_new_locs
                ),
                lambda: all_new_locs
            )

            # collect apples and update inventory accordingly (note that the apples get removed implicitly when we update the grid
            #as the agents position overides the collected apple's position)
            def coin_matcher(p: jnp.ndarray) -> jnp.ndarray:
                c_matches = jnp.array([
                    state.grid[p[0], p[1]] == Items.apple
                    ])
                return c_matches
            
            apple_matches = jax.vmap(coin_matcher)(p=new_locs)

            new_invs = state.agent_invs + apple_matches
            
            # Update cumulative apple collection
            new_cumulative_apples = state.cumulative_apples_collected + apple_matches.squeeze()

            state = state.replace(
                agent_invs=new_invs,
                cumulative_apples_collected=new_cumulative_apples
            )

            # update grid
            old_grid = state.grid

            new_grid = old_grid.at[
                state.agent_locs[:, 0],
                state.agent_locs[:, 1]
            ].set(
                jnp.int16(Items.empty)
            )

            new_grid = new_grid.at[state.potential_dirt_and_dirt_locs[:, 0], state.potential_dirt_and_dirt_locs[:, 1]].set(state.potential_dirt_and_dirt_label)

            new_grid = new_grid.at[self.RIVER[:, 0], self.RIVER[:, 1]].set(Items.river)
            x, y = new_locs[:, 0], new_locs[:, 1]
            new_grid = new_grid.at[x, y].set(self._agents)
            state = state.replace(grid=new_grid)

            # update agent locations
            state = state.replace(agent_locs=new_locs)

            reborn_players, state = _interact_fire_zapping(key, state, actions)

            state = _interact_fire_cleaning(key, state, actions)

            reborn_players_3d = jnp.stack([reborn_players, reborn_players, reborn_players], axis=-1)

            # jax.debug.print("reborn_players_3d {reborn_players_3d} 🤯", reborn_players_3d=reborn_players_3d)

            def respawn_branch(_):
                re_agents_pos = jax.random.permutation(k_respawn_perm, self.SPAWNS_PLAYERS)[:num_agents]
                player_dir = jax.random.randint(
                    k_respawn_dir, shape=(num_agents,), minval=0, maxval=3, dtype=jnp.int8
                )
                re_agent_locs = jnp.array(
                    [re_agents_pos[:, 0], re_agents_pos[:, 1], player_dir],
                    dtype=jnp.int16
                ).T
                new_re_locs_inner = jnp.where(~reborn_players_3d, new_locs, re_agent_locs)
                return new_re_locs_inner

            def no_respawn_branch(_):
                return state.agent_locs

            new_re_locs = jax.lax.cond(reborn_players.any(), respawn_branch, no_respawn_branch, operand=None)
            state = state.replace(reborn_locs=new_re_locs)

            # Calculate base rewards...i.e. if agent collected apple add 1 to its respective reward
            base_rewards = jnp.zeros((self.num_agents, 1), dtype=jnp.bfloat16)
            base_apple_rewards = jnp.where(apple_matches, 1, base_rewards)
            
            # REWARD LOGIC:
            # =============

            # Apply reward type logic
            if self.reward_type == "shared":
                # Shared rewards: all agents get sum of all rewards
                rewards_sum_all_agents = jnp.zeros((self.num_agents, 1), dtype=jnp.bfloat16)
                rewards_sum = jnp.sum(base_apple_rewards)
                rewards_sum_all_agents += rewards_sum
                rewards = rewards_sum_all_agents
                info = {
                    "individual_rewards": base_apple_rewards.squeeze(),
                }
            elif self.reward_type == "individual":
                # Individual rewards: each agent gets only their own rewards
                rewards = base_apple_rewards
                info = {"individual_rewards": rewards.squeeze(),}
            elif self.reward_type == "saturating":
                # Saturating rewards: only penalize agents with maximum apples
                # Get current cumulative apple counts for all agents
                agent_apple_counts = state.cumulative_apples_collected
                
                # Find the maximum apple count
                max_apples = jnp.max(agent_apple_counts)

                coef = jax.lax.cond(
                    max_apples == 0,
                    lambda _: jnp.ones_like(agent_apple_counts, dtype=jnp.bfloat16),
                    lambda _: jnp.asarray(1.0, dtype=jnp.bfloat16) - jnp.asarray(agent_apple_counts, dtype=jnp.bfloat16) / jnp.asarray(max_apples, dtype=jnp.bfloat16) + jnp.asarray(0.2, dtype=jnp.bfloat16),
                    operand=max_apples,
                )
                
                # # Create mask for agents with maximum apples (handle ties)
                # has_max_apples = agent_apple_counts == max_apples
                
                # # Only agents with max apples get zero reward, all others get full reward
                # reward_multipliers = jnp.where(has_max_apples, 0.0, 1.0).reshape(-1, 1)
                saturated_apple_rewards = base_apple_rewards * coef[...,jnp.newaxis]

                
                rewards = saturated_apple_rewards
                info = {"individual_rewards": rewards.squeeze(),}
            elif self.reward_type == "fractional":
                # Fractional rewards: agents get individual rewards + 0.1 * team total
                team_total_reward = jnp.sum(base_apple_rewards)
                fractional_team_bonus = jnp.asarray(0.5, dtype=jnp.bfloat16) * team_total_reward
                
                # Each agent gets their individual reward + fractional team bonus
                fractional_rewards = base_apple_rewards + fractional_team_bonus
                rewards = fractional_rewards
                info = {"individual_rewards": base_apple_rewards.squeeze(),}
            else:
                raise ValueError(f"Invalid reward_type: '{self.reward_type}'. Must be 'shared', 'individual', 'saturating', or 'fractional'.")
            
            info["clean_action_info"] = jnp.where(actions == Actions.zap_clean, 1, 0).squeeze()
            info["cleaned_water"] = jnp.array([len(state.potential_dirt_and_dirt_label) - dirtCount] * self.num_agents).squeeze()
            info["cumulative_apples_collected"] = state.cumulative_apples_collected.squeeze()
            
            state_nxt = State(
                agent_locs=state.agent_locs,
                agent_invs=state.agent_invs,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                apples=state.apples,
                freeze=state.freeze,
                reborn_locs=state.reborn_locs,
                potential_dirt_and_dirt_locs=state.potential_dirt_and_dirt_locs,
                potential_dirt_and_dirt_label=state.potential_dirt_and_dirt_label,
                smooth_rewards=state.smooth_rewards,
                cumulative_apples_collected=state.cumulative_apples_collected
            )

            # now calculate if done for inner or outer episode
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            reset_inner = inner_t == num_inner_steps

            # if inner episode is done, return start state for next game
            state_re = _reset_state(key)

            state_re = state_re.replace(
                outer_t=outer_t + 1, 
                cumulative_apples_collected=state_nxt.cumulative_apples_collected
            )
            state = jax.tree_map( #resets the entire state
                lambda x, y: jnp.where(reset_inner, x, y),
                state_re,
                state_nxt,
            )
            outer_t = state.outer_t
            reset_outer = outer_t == num_outer_steps
            done = {f'{a}': reset_outer for a in self.agents}
            # done = [reset_outer for _ in self.agents]
            done["__all__"] = reset_outer

            obs = _get_obs(state)
            rewards = jnp.where(
                reset_inner,
                jnp.zeros_like(rewards),
                rewards
            )

            # mean_inv = state.agent_invs.mean(axis=0)
            return (
                obs,
                state,
                rewards.squeeze(),
                done,
                info,
            )

        def _reset_state(
            key: jnp.ndarray
        ) -> State:
            key, subkey = jax.random.split(key)

            # Find the free spaces in the grid
            grid = jnp.zeros((self.GRID_SIZE_ROW, self.GRID_SIZE_COL), jnp.int16)


            inside_players_pos = jax.random.permutation(subkey, self.SPAWNS_PLAYER_IN)
            player_positions = jnp.concatenate((inside_players_pos, self.SPAWNS_PLAYERS))
            agent_pos = jax.random.permutation(subkey, player_positions)[:num_agents]
            wall_pos = self.SPAWNS_WALL
            apple_pos = self.POTENTIAL_APPLE

            river = self.RIVER
            potential_dirt = self.POTENTIAL_DIRT
            dirt = self.DIRT

            potential_dirt_label = jnp.zeros((len(potential_dirt)), dtype=jnp.int16) +Items.potential_dirt
            dirt_label = jnp.zeros((len(dirt)), dtype=jnp.int16) + Items.dirt

            potential_dirt_and_dirt = jnp.concatenate((potential_dirt, dirt))
            potential_dirt_and_dirt_label = jnp.concatenate((potential_dirt_label, dirt_label))


            # set wall
            grid = grid.at[
                wall_pos[:, 0],
                wall_pos[:, 1]
            ].set(jnp.int16(Items.wall))

            # set dirt
            grid = grid.at[dirt[:, 0],
                           dirt[:, 1]
                           ].set(jnp.int16(Items.dirt))
            
            # set river
            grid = grid.at[river[:, 0],
                            river[:, 1]
                            ].set(jnp.int16(Items.river))
            
            # set potential dirt
            grid = grid.at[potential_dirt[:, 0],
                            potential_dirt[:, 1]
                            ].set(jnp.int16(Items.potential_dirt))
            


            player_dir = jax.random.randint(
                subkey, shape=(
                    num_agents,
                    ), minval=0, maxval=3, dtype=jnp.int8
            )

            agent_locs = jnp.array(
                [agent_pos[:, 0], agent_pos[:, 1], player_dir],
                dtype=jnp.int16
            ).T

            grid = grid.at[
                agent_locs[:, 0],
                agent_locs[:, 1]
            ].set(jnp.int16(self._agents))

            freeze = jnp.array(
                [[-1]*num_agents]*num_agents,
            dtype=jnp.int16
            )

            return State(
                agent_locs=agent_locs,
                agent_invs=jnp.array([(0,0)]*num_agents, dtype=jnp.int8),
                inner_t=0,
                outer_t=0,
                grid=grid,
                apples=apple_pos,

                freeze=freeze,
                reborn_locs=agent_locs,
                potential_dirt_and_dirt_locs=potential_dirt_and_dirt,
                potential_dirt_and_dirt_label=potential_dirt_and_dirt_label,
                smooth_rewards=jnp.zeros((self.num_agents, 1)),
                cumulative_apples_collected=jnp.zeros(self.num_agents, dtype=jnp.int32)
            )

        def reset(
            key: jnp.ndarray
        ):
            state = _reset_state(key)
            obs = _get_obs(state)
            return obs, state
        
        ################################################################################
        # if you want to test whether it can run on gpu, activate following code
        # overwrite Gymnax as it makes single-agent assumptions
        if jit:
            self.step_env = jax.jit(_step)
            self.reset = jax.jit(reset)
            self.get_obs_point = jax.jit(_get_obs_point)
        else:
            # if you want to see values whilst debugging, don't jit
            self.step_env = _step
            self.reset = reset
            self.get_obs_point = _get_obs_point
        ################################################################################

    @property
    def name(self) -> str:
        """Environment name."""
        return "MGinTheGrid"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Actions)

    def action_space(
        self, agent_id: Union[int, None] = None
    ) -> Discrete:
        """Action space of the environment."""
        return Discrete(len(Actions))

    def observation_space(self) -> Dict:
        """Observation space of the environment."""
        # Base channels: (len(Items)-1) + 10, plus coefficient channel (1), plus agent ID channels if enabled
        base_channels = (len(Items)-1) + 10
        coef_channels = 1  # Always add coefficient channel
        agent_id_channels = self.num_agents if self.agent_ids else 0
        total_channels = base_channels + coef_channels + agent_id_channels
        
        _shape_obs = (
            (self.OBS_SIZE, self.OBS_SIZE, total_channels)
            if self.cnn
            else (self.OBS_SIZE**2 * total_channels,)
        )

        return Box(
                low=0, high=1E9, shape=_shape_obs, dtype=jnp.uint8
            ), _shape_obs
    
    def state_space(self) -> Dict:
        """State space of the environment."""
        _shape = (
            (self.GRID_SIZE_ROW, self.GRID_SIZE_COL, NUM_TYPES + 4)
            if self.cnn
            else (self.GRID_SIZE_ROW* self.GRID_SIZE_COL * (NUM_TYPES + 4),)
        )
        return Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)
    
    def render_tile(
        self,
        obj: int,
        agent_dir: Union[int, None] = None,
        agent_hat: bool = False,
        highlight: bool = False,
        tile_size: int = 32,
        subdivs: int = 3,
    ) -> onp.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, agent_hat, highlight, tile_size)
        if obj:
            key = (obj, 0, 0, 0) + key if obj else key

        if key in self.tile_cache:
            return self.tile_cache[key]

        img = onp.full(
                shape=(tile_size * subdivs, tile_size * subdivs, 3),
                fill_value=(190, 170, 120),
                dtype=onp.uint8,
            )

    # class Items(IntEnum):

        if obj in self._agents:
            # Draw the agent
            agent_color = self.PLAYER_COLOURS[obj-len(Items)]
        elif obj == Items.apple:
            # Draw the red coin as GREEN COOPERATE
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (214.0, 39.0, 40.0)
            )
        
        # elif obj == Items.blue_coin:
        #     # Draw the blue coin as DEFECT/ RED COIN
        #     fill_coords(
        #         img, point_in_circle(0.5, 0.5, 0.31), (214.0, 39.0, 40.0)
        #     )

        elif obj == Items.river:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (40.0, 80.0, 214.0))
        elif obj == Items.potential_dirt:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (40.0, 80.0, 214.0))
        elif obj == Items.dirt:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (40.0, 80.0, 80.0))


        elif obj == Items.wall:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (127.0, 127.0, 127.0))

        elif obj == Items.interact:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (188.0, 189.0, 34.0))

        elif obj == Items.clean_beam:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (170, 220, 255))
            print(Items.clean_beam)

        elif obj == 99:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (44.0, 160.0, 44.0))

        elif obj == 100:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (214.0, 39.0, 40.0))

        elif obj == 101:
            # white square
            fill_coords(img, point_in_rect(0, 1, 0, 1), (255.0, 255.0, 255.0))

        # Overlay the agent on top
        if agent_dir is not None:
            if agent_hat:
                tri_fn = point_in_triangle(
                    (0.12, 0.19),
                    (0.87, 0.50),
                    (0.12, 0.81),
                    0.3,
                )

                # Rotate the agent based on its direction
                tri_fn = rotate_fn(
                    tri_fn,
                    cx=0.5,
                    cy=0.5,
                    theta=0.5 * math.pi * (1 - agent_dir),
                )
                fill_coords(img, tri_fn, (255.0, 255.0, 255.0))

            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
                0.0,
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (1 - agent_dir)
            )
            fill_coords(img, tri_fn, agent_color)

        # # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        self.tile_cache[key] = img
        return img

    def render(
        self,
        state: State,
    ) -> onp.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        tile_size = 32
        highlight_mask = onp.zeros_like(onp.array(self.GRID))

        # Compute the total grid size
        width_px = self.GRID.shape[1] * tile_size
        height_px = self.GRID.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)

        grid = onp.array(state.grid)
        # print(onp.argwhere(grid == Items.clean_beam))
        grid = onp.pad(
            grid, ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)), constant_values=Items.wall
        )
        for a in range(self.num_agents):
            startx, starty = self.get_obs_point(
                state.agent_locs[a]
            )
            highlight_mask[
                startx : startx + self.OBS_SIZE, starty : starty + self.OBS_SIZE
            ] = True

        # Render the grid
        for j in range(0, grid.shape[1]):
            for i in range(0, grid.shape[0]):
                cell = grid[i, j]
                if cell == 0:
                    cell = None
                agent_here = []
                for a in self._agents:
                    agent_here.append(cell == a)
                # if cell in [1,2]:
                #     print(f'coordinates: {i},{j}')
                #     print(cell)

                agent_dir = None
                for a in range(self.num_agents):
                    agent_dir = (
                        state.agent_locs[a,2].item()
                        if agent_here[a]
                        else agent_dir
                    )
                
                agent_hat = False
                # for a in range(self.num_agents):
                #     agent_hat = (
                #         bool(state.agent_invs[a].sum() > INTERACT_THRESHOLD)
                #         if agent_here[a]
                #         else agent_hat
                #     )

                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir,
                    agent_hat=agent_hat,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = i * tile_size
                ymax = (i + 1) * tile_size
                xmin = j * tile_size
                xmax = (j + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
        
        img = onp.rot90(
            img[
                (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
                (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
                :,
            ],
            2,
        )
        # time = self.render_time(state, img.shape[1])
        # img = onp.concatenate((img, time), axis=0)
        return img



    def render_time(self, state, width_px) -> onp.array:
        inner_t = state.inner_t
        outer_t = state.outer_t
        tile_height = 32
        img = onp.zeros(shape=(2 * tile_height, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // (self.num_inner_steps)
        j = 0
        for i in range(0, inner_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        tile_width = width_px // (self.num_outer_steps)
        j = 1
        for i in range(0, outer_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        return img
    
# ISP ENVIRONMENT 

class ISP(MultiAgentEnv):

    title_cache = {}

    def __init__(
            self,
            num_agents=3,
            num_inner_steps=200, # length of episode
            num_outer_steps=1, # number of episodes ran
            # river regeneration hyperparams (a.k.a. ecological hyperparams):
            # EQUATION for ease of reference: R_{t+1} = clip(R_t + alpha.R_t(1-R_t) - DD_t + I_t + eps_t, 0, 1)
            alpha=0.3, # intrinsic logistic regen rate 
            beta_h=0.1, # energy gain per harvest action
            gamma_h=0.1, # river damage per harvest action for (D_t,i in regen function)
            beta_v = 0.2, # energy loss per invest action
            gamma_v=0.3, # river health increase per invest (for I_t in regen function)
            g = 0.01, # NoOp energy gain for agent not doing anything at that specific number step
            K_collapse_thresh = 0.1, # river health level collapse threshold (after k steps below threshold, episode transitions into terminal state)
            k_collapse_steps = 5, # number of steps below K before episode collapses
            sigma_noise = 0.05, # standard deviation for Normal distribution for noise
            #reward function hyperparams:
            w_f=1.0, # weight for delta energy
            w_h = 5.0, # weight for Indicator function for agent being below survival threshold...if agent is below threshold - huge penalty
            w_c = 1.0, # weight for indicator function for river health being below K collapse shreshold
            w_p=0.1, # weight for indicator function for agent being punished
            lambda_h = 0.1, # hunger threshold for energy level...if energy level drops below this, agent is penalised heavily with w_h
            #lambda_c=0.2, # river collapse hyperparam...we shall start with K as this value, but might provide interesting dynamics to have different K values with lambdda_c to start penalising agents earlier rather than later
            c_pun=0.08, # energy cost for agent to punish someone else
            c_rec=0.16, # energy cost for agent who is receiving punishment
            jit = True,
            obs_size=11, # each agent has fov of 11x11 tiles within grid
            cnn=True,
            agent_ids=False,
            # This is a 15x9 gridworld which we have proposed, on the basis that we will use 3 agents to start out. There are walls on the left and right and bottom of the environment to prevent the agents walking out. 
            # the reason for the wall between the river and the agents living area is to prevent an agent just being able stand next to a river for infinite time and continue harvesting/investing...they actually need to make a choice to leave the living area and enter the river to harvest/invest
            map_ASCII = [
                    'WRRRRRRRRRRRRRW',
                    'WRRRRRRRRRRRRRW',
                    'WRRRRRRRRRRRRRW',
                    'WWWWWWWWWWWWWWW',
                    '   P    P    P ',
                    '  P    P    P  ',
                    '   P    P    P ',
                    'WWWWWWWWWWWWWWW',
                ]
    ):
        super().__init__(num_agents=num_agents)

        self.alpha=alpha
        self.beta_h = beta_h
        self.beta_v = beta_v
        self.gamma_h=gamma_h
        self.gamma_v = gamma_v
        self.g = g
        self.K_collapse_thresh = K_collapse_thresh
        self.k_collapse_steps= k_collapse_steps
        self.sigma_noise = sigma_noise
        self.w_f = w_f
        self.w_h = w_h
        self.w_c = w_c
        self.w_p = w_p
        self.lambda_h = lambda_h
        self.c_pun = c_pun
        self.c_rec = c_rec
        self.cnn = cnn
        self.agent_ids = agent_ids
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps

        self.agents = list(range(num_agents)) # indexing agents 
        self._agents = jnp.array(self.agents, dtype=jnp.int16) + len(Items) # we need to offset agents by number of items we have in the grid so we know what each tile within the grid contains

        self.PLAYER_COLOURS = generate_agent_colors(num_agents) # agent rendering colours

        # grid dimensions are as follows:
        self.GRID_SIZE_ROW = len(map_ASCII)
        self.GRID_SIZE_COL = max(len(row) for row in map_ASCII)
        self.OBS_SIZE = obs_size
        self.PADDING = self.OBS_SIZE -1 

        GRID = jnp.zeros(
            (self.GRID_SIZE_ROW + 2 * self.PADDING, self.GRID_SIZE_COL + 2*self.PADDING),
            dtype=jnp.int16,
        )
        GRID = GRID.at[self.PADDING - 1, :].set(Items.wall)
        GRID = GRID.at[:, self.PADDING - 1].set(Items.wall)
        self.GRID = GRID.at[:, self.GRID_SIZE_COL + self.PADDING].set(Items.wall)

        def find_positions(grid_array, value):
            return jnp.array(jnp.where(grid_array == value)).T
        
        nums_map = ascii_map_to_matrix(map_ASCII, char_to_int)
        self.RIVER = find_positions(nums_map, char_to_int['R']) # river tiles...i.e. harvest/invest zone
        self.SPAWNS_PLAYERS = find_positions(nums_map, char_to_int['P']) # agent spawn locations
        self.SPAWNS_WALL = find_positions(nums_map, char_to_int['W']) # wall tiles

        # Define Punish(j) action. Note that '9+j', means "punish agent j":
        self.PUNISH_ACTIONS = {j: 9+j for j in range(num_agents)}

        # from above, we add the dynamics for punish(j) for rotations and step_move, which do not move or rotate any agent when punishing an agent
        punish_rows = jnp.zeros((num_agents,3), dtype = jnp.int8)
        self.ROTATIONS = jnp.concatenate([ROTATIONS,punish_rows], axis=0)
        self.STEP_MOVE = jnp.concatenate([STEP_MOVE, punish_rows], axis=0)


        # Collision logic from CLEANUP: 
        def check_collision(
                new_agent_locs: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function to check agent collisions.
            
            Args:
                - new_agent_locs: jnp.ndarray, the agent locations at the 
                current time step.
                
            Returns:
                - jnp.ndarray matrix of bool of agents in collision.
            '''
            matcher = jax.vmap(
                lambda x,y: jnp.all(x[:2] == y[:2]),
                in_axes=(0, None)
            )

            collisions = jax.vmap(
                matcher,
                in_axes=(None, 0)
            )(new_agent_locs, new_agent_locs)

            return collisions
        
        def fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Function defining multi-collision logic.
            If agents move into tile, and there is already and agent in that tile,
            then agent who was there has right of way, and the other agents get reset to where they were.
            Case 2: if all agents moved to a specfic tile, then one will win at random, and the others who lose out
            on the tile will be reset to their previous tile.

            Args:
                - key: jax key for randomisation
                - collided_moved: jnp.ndarray, the agents which moved in the
                last time step and caused collisions.
                - collision_matrix: jnp.ndarray, the agents currently in
                collisions
                - agent_locs: jnp.ndarray, the agent locations at the previous
                time step.
                - new_agent_locs: jnp.ndarray, the agent locations at the
                current time step.

            Returns:
                - jnp.ndarray of the final positions after collisions are
                managed.
            """
            def scan_fn(
                    state,
                    idx
            ):
                key, collided_moved, collision_matrix, agent_locs, new_agent_locs = state

                return jax.lax.cond(
                    collided_moved[idx] > 0,
                    lambda: _fix_collisions(
                        key,
                        collided_moved,
                        collision_matrix,
                        agent_locs,
                        new_agent_locs
                    ),
                    lambda: (state, new_agent_locs)
                )

            _, ys = jax.lax.scan(
                scan_fn,
                (key, collided_moved, collision_matrix, agent_locs, new_agent_locs),
                jnp.arange(self.num_agents)
            )

            final_locs = ys[-1]

            return final_locs

        def _fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ):
            def select_random_true_index(key, array):
                # Calculate the cumulative sum of True values
                cumsum_array = jnp.cumsum(array)

                # Count the number of True values
                true_count = cumsum_array[-1]

                # Generate a random index in the range of the number of True
                # values
                rand_index = jax.random.randint(
                    key,
                    (),
                    0,
                    true_count
                )

                # Find the position of the random index within the cumulative
                # sum
                chosen_index = jnp.argmax(cumsum_array > rand_index)

                return chosen_index
            # Pick one from all who collided & moved
            colliders_idx = jnp.argmax(collided_moved)

            collisions = collision_matrix[colliders_idx]

            # Check whether any of collision participants didn't move
            collision_subjects = jnp.where(
                collisions,
                collided_moved,
                collisions
            )
            collision_mask = collisions == collision_subjects
            stayed = jnp.all(collision_mask)
            stayed_mask = jnp.logical_and(~stayed, ~collision_mask)
            stayed_idx = jnp.where(
                jnp.max(stayed_mask) > 0,
                jnp.argmax(stayed_mask),
                0
            )

            # Prepare random agent selection
            k1, k2 = jax.random.split(key, 2)
            rand_idx = select_random_true_index(k1, collisions)
            collisions_rand = collisions.at[rand_idx].set(False)
            new_locs_rand = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_rand,
                agent_locs,
                new_agent_locs
            )

            collisions_stayed = jax.lax.select(
                jnp.max(stayed_mask) > 0,
                collisions.at[stayed_idx].set(False),
                collisions_rand
            )
            new_locs_stayed = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_stayed,
                agent_locs,
                new_agent_locs
            )

            # Choose between the two scenarios - revert positions if
            # non-mover exists, otherwise choose random agent if all moved
            new_agent_locs = jnp.where(
                stayed,
                new_locs_rand,
                new_locs_stayed
            )

            # Update move bools to reflect the post-collision positions
            collided_moved = jnp.clip(collided_moved - collisions, 0, 1)
            collision_matrix = collision_matrix.at[colliders_idx].set(
                [False] * collisions.shape[0]
            )
            return ((k2, collided_moved, collision_matrix, agent_locs, new_agent_locs), new_agent_locs)


        def _get_obs_point(agent_loc: jnp.ndarray) -> jnp.array:
            '''
            Obtain the position of top-left corner of obs map using
            agent's current location & orientation.

            Args: 
                - agent_loc: jnp.ndarray, agent x, y, direction.
            Returns:
                - x, y: ints of top-left corner of agent's obs map.
            '''
            x, y, direction = agent_loc
            x, y = x + self.PADDING, y + self.PADDING
            x = x - (self.OBS_SIZE // 2)
            y = y - (self.OBS_SIZE // 2)
            x = jnp.where(direction == 0, x + (self.OBS_SIZE // 2) - 1, x)
            x = jnp.where(direction == 2, x - (self.OBS_SIZE // 2) + 1, x)
            y = jnp.where(direction == 1, y + (self.OBS_SIZE // 2) - 1, y)
            y = jnp.where(direction == 3, y - (self.OBS_SIZE // 2) + 1, y)
            return x, y
        
        def rotate_grid(agent_loc: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
            '''
            Rotates agent's observation grid by 0/90/180/270 degrees based on direction.
            Uses a data-dependent branch to avoid computing all rotations. 
            This aligns the world to that specific agent's facing-direction (egocentric), so that the CNN policy
            training becomes dramatically easier.
            '''
            def rot0(g): return g
            def rot1(g): return jnp.rot90(g, k=1, axes=(0, 1))
            def rot2(g): return jnp.rot90(g, k=2, axes=(0, 1))
            def rot3(g): return jnp.rot90(g, k=3, axes=(0, 1))
            return jax.lax.switch(jnp.asarray(agent_loc[2], jnp.int32), [rot0, rot1, rot2, rot3], grid)
        
        def _get_obs(state: State) -> jnp.ndarray:
            '''
            Obtain the agent's observation of the grid.

            Args: 
                - state: State object containing env state.
            Returns:
                - jnp.ndarray of grid observation.
            '''
            # pad grid with walls for out-of-bounds cells
            grid = jnp.pad(
                state.grid,
                ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)),
                constant_values=Items.wall,
            )
            # get top-left corner of each agent's obs window
            agent_start_idxs = jax.vmap(_get_obs_point)(state.agent_locs)

            dynamic_slice = partial(
                jax.lax.dynamic_slice,
                operand=grid,
                slice_sizes=(self.OBS_SIZE, self.OBS_SIZE),
            )

            # slice and rotate to egocentric frame
            grids = jax.vmap(dynamic_slice)(start_indices=agent_start_idxs)
            grids = jax.vmap(rotate_grid)(state.agent_locs, grids)

            # one-hot encode so the categories are (empty, wall, river, agent_0, agent_1, ...)
            # shift by -1 so empty(0)->-1 gets zeroed out by one_hot...because it contains no additional info, wall(1)->0, etc.
            # num classes = len(Items) - 1 + num_agents  (drop empty block and keep wall, river, agents)
            num_classes = len(Items) - 1 + num_agents
            grids = jax.nn.one_hot(grids - 1, num_classes, dtype=jnp.int8)
            # grids shape: (num_agents, OBS_SIZE, OBS_SIZE, num_classes)

            
            def add_scalar_channels(grid,agent_idx):
                '''
                add river channel and energy channel to one-hot encoded vector, which we will later pass into CNN in IPPO...
                !Note that here we just add the 2 channels as part of the whole channel, but CNN should learn to recognise these are constants thus ignore. 
                When we add GRU with communication action, this will have to change
                '''
                # noisy river estimate for this agent: shape (OBS_SIZE, OBS_SIZE, 1)
                river_ch = jnp.full(
                    (self.OBS_SIZE, self.OBS_SIZE, 1),
                    state.river_obs[agent_idx],
                    dtype=jnp.float32,
                )
                # agent's own energy: shape (OBS_SIZE, OBS_SIZE, 1)
                energy_ch = jnp.full(
                    (self.OBS_SIZE, self.OBS_SIZE, 1),
                    state.energy[agent_idx],
                    dtype=jnp.float32,
                )
                return jnp.concatenate([grid, river_ch, energy_ch], axis=-1)
            
            grids = jax.vmap(add_scalar_channels, in_axes=(0,0)) (
                grids.astype(jnp.float32), jnp.arange(num_agents) # cast grid to float 32, since it is int8
            )
            return grids # shape is: (num_agents, OBS_SIZE, OBS_SIZE, num_classes + 2...for river channel and energy channel included)
        
        def _step(
            key: chex.PRNGKey,
            state: State,
            actions: jnp.ndarray
        ):
            """Step the ISP environment:"""

            # split the keys upfront so the different noise sources are independent...river noise needs to be independent from observation noise, so agents do not infer true river state from noise structure
            # Moreover, we do not want collision key to be correlated with river noise or obs noise, otherwise agents could use collision dynamics to infer state of river/observation of other agents
            key, k_river_noise, k_obs_noise, k_collision = jax.random.split(key, 4)

            actions = jnp.array(actions)

            grid = state.grid.at[
                # clear agents from the grid first:
                state.agent_locs[:, 0],
                state.agent_locs[:, 1]
            ].set(jnp.int16(Items.empty))
            grid = grid.at[self.RIVER[:, 0], self.RIVER[:, 1]].set(jnp.int16(Items.river)) # explictly add back the river tiles
            state = state.replace(grid=grid) 

            # Movement Logic for Step Function: 

            all_new_locs = jax.vmap(
                lambda p, a: jnp.int16(p+self.ROTATIONS[a]) % jnp.array( # rotate all agents at once 
                    [self.GRID_SIZE_ROW + 1, self.GRID_SIZE_COL+1, 4], dtype=jnp.int16
                )
            )(state.agent_locs, actions).squeeze()

            agent_move = ( # boolean mask to update agents' locations in the next step, given that they chose a movement action
                (actions == Actions.up) | (actions == Actions.down) | 
                (actions == Actions.left) | (actions == Actions.right)
            )
            all_new_locs = jax.vmap( # move all the agents accordingly to their new respective locations
                lambda m, n, a: jnp.where(m,n+self.STEP_MOVE[a], n)
            )(agent_move, all_new_locs, actions)

            all_new_locs = jax.vmap( # make sure that the agent's move does not cause it to move off the grid...thus, clip its bounds to prevent further handling
                jnp.clip, in_axes = (0, None, None)
            )(
                all_new_locs,
                jnp.array([0,0,0], dtype=jnp.int16),
                jnp.array([self.GRID_SIZE_ROW-1, self.GRID_SIZE_COL-1, 3], dtype=jnp.int16)
            ).squeeze()

            # Collision Logic for Step Function: 

            agents_moved = jax.vmap( # for all agents compare proposed new location with old location and check if agent proposed to move. If they do change position, return true 
                lambda n, p: jnp.any(n[:2] != p[:2])
            )(all_new_locs, state.agent_locs)

            collision_matrix = check_collision(all_new_locs) # setup matrix for collision logic...matrix of Bool values - if agent i and j are proposing to occupy same tile then set to true
            collisions = jnp.minimum(
                jnp.sum(collision_matrix, axis=-1, dtype=jnp.int8) -1, 1 # count number of agents who share the same tile, and subtract one for the self collision. Then clamp the value to 1, since we only care if there was a collision or not
            )
            collided_moved = jnp.maximum(collisions -~agents_moved, 0) # return 1 if the agent moved and caused a collision

            new_locs = jax.lax.cond(
                jnp.max(collided_moved) > 0,
                lambda: fix_collisions( # if there is a collision, then fix the collision according to the fix_collisions logic
                    k_collision, collided_moved, collision_matrix,
                    state.agent_locs, all_new_locs
                ),
                lambda: all_new_locs
            )

            def is_on_river(loc):
                return jnp.any(jnp.all(self.RIVER == loc[:2], axis=-1)) # check to see whether an agent is standing on a river tile...recall agent cannot harvest or invest unless on the river
            on_river = jax.vmap(is_on_river)(new_locs)


            # classify agent actions to update energy levels, cumulative harvest and cumulative invest, c_pun, c_rec, w_p to reward
            harvesting = (actions == Actions.harvest) & on_river
            investing = (actions == Actions.invest) & on_river
            noop = (actions == Actions.stay)
            punishing = actions >= 9 # Recall Punish(j) = 9 + j
            punish_target = jnp.clip(actions-9, 0, num_agents-1)

            # Update per agent energy levels: 
            energy_old = state.energy # vector with per agent energy levels

            energy_delta = ( # this is delta_energy for self actions
                jnp.where(harvesting, self.beta_h, 0.0)
                + jnp.where(investing, -self.beta_v, 0.0)
                + jnp.where(noop, self.g, 0.0)
                + jnp.where(punishing, -self.c_pun, 0.0)
            ) 

            def punishment_received(j): # also need to account for agents how received total punishment from other agents within env
                return jnp.sum(
                    jnp.where(punishing & (punish_target == j), self.c_rec, 0.0)
                )
            energy_delta = energy_delta - jax.vmap(punishment_received)(jnp.arange(num_agents)) # update energy levels of each agent by taking delta_energy_self and subtracting energy loss from receiving punishment per agent
            
            energy_new = jnp.clip(energy_old + energy_delta, 0.0, 1.0) 

            # River Dynamics Update: recall R_{t+1} = clip(R_t + alpha.R_t(1-R_t) - D_t = I_t + eps_t, 0, 1)
            D_t = jnp.sum(jnp.where(harvesting, self.gamma_h, 0.0)) # damage to be subtracted from river health from agents harvesting
            I_t = jnp.sum(jnp.where(investing, self.gamma_v, 0.0)) # health boost to be added back to river from agents investing 
            eps_t = jax.random.normal(k_river_noise)* self.sigma_noise # eps_t ~ N(0, sigma^2)

            R_t = state.river_level
            R_new = jnp.clip( # update river level according to logistic regeneration process abot
                R_t + self.alpha*R_t*(1.0-R_t) - D_t + I_t + eps_t, 
                0.0, 1.0
            )

            below_K = R_new < self.K_collapse_thresh # bool to check if river health level is below collapse threshold
            num_steps_below = jnp.where( # if the river health is below the threshold, then increment the count by one... remember that if the count goes above our threshold then environment reaches a terminal state
                below_K,
                state.num_steps_below_collapse + 1,
                jnp.int32(0)
            )

            obs_noise = jax.random.normal(k_obs_noise, shape=(num_agents,))* self.sigma_noise
            river_obs_new = jnp.clip(R_new + obs_noise, 0.0, 1.0) # agents make local noisy observations about the environment each step 

            # update the cumulative harvest and invest amount to be used in the greed metric: 
            new_cumulative_harvest = state.cumulative_harvest + harvesting.astype(jnp.float32)
            new_cumulative_invest = state.cumulative_invest + investing.astype(jnp.float32)

            # Reward function Update: Recall reward is: u_{t,i} = w_f*delta_e - w_h.I[e<=lambda_h] - w_c*I[r<=k] - w_p*[punish]
            delta_e = energy_new - energy_old
            hunger_penalty = jnp.where(energy_new <= self.lambda_h, self.w_h, 0.0)
            collapse_penalty = jnp.where(R_new<=self.K_collapse_thresh, self.w_c, 0.0)
            punish_penalty = jnp.where(punishing, self.w_p, 0.0)
            rewards = (self.w_f*delta_e) - hunger_penalty - collapse_penalty - punish_penalty # reward function

            # update the grid with river tiles and place agents on their new respective tiles: 
            new_grid = state.grid.at[self.RIVER[:, 0], self.RIVER[:, 1]].set(jnp.int16(Items.river))
            new_grid = new_grid.at[new_locs[:, 0], new_locs[:, 1]].set(self._agents)

            state_nxt = State( # build out the next state at time t+1
                agent_locs=new_locs,
                river_level=R_new,
                energy=energy_new,
                reputations=state.reputations, # we haven't changed this yet...once adding audit logic then we will change reputations vector
                cumulative_harvest=new_cumulative_harvest,
                cumulative_invest=new_cumulative_invest,
                num_steps_below_collapse=num_steps_below,
                river_obs=river_obs_new,
                grid=new_grid,
                inner_t = state.inner_t+1,
                outer_t=state.outer_t,
            )

            # episode reset logic:
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            collapse_done = num_steps_below>=self.k_collapse_steps # reset if the number of steps that river is below collapse threshold reaches number of collapse steps
            reset_inner = (inner_t == num_inner_steps) | collapse_done # also reset if reached total number of time steps specified within the episode

            state_re = _reset_state(key)
            state_re = state_re.replace(outer_t=outer_t+1)

            state = jax.tree_map( # if the episode is terminated, then reset, otherwise, go to the next state
                lambda x, y: jnp.where(reset_inner, x, y),
                state_re,
                state_nxt
            )

            # check to see that training is done, if so then give done flag
            outer_t = state.outer_t 
            reset_outer = outer_t == num_outer_steps
            done = {f'{a}': reset_outer for a in self.agents}
            done["__all__"] = reset_outer

            obs = _get_obs(state)
            rewards = jnp.where(reset_inner, jnp.zeros_like(rewards), rewards) # reset rewards after episode complete

            info = { # return some diagnostics which we can go through for our own reference...this is not for the agentss' training
                "harvest": harvesting.astype(jnp.float32),
                "invest": investing.astype(jnp.float32),
                "punish": punishing.astype(jnp.float32),
                "river_level": R_new,
                "energy": energy_new,
                "collapse": collapse_done,
            }

            return obs, state, rewards, done, info
        
        # reset function which also initialises the state of the episode:
        def _reset_state(
                key: jnp.ndarray
        ) -> State:
            key, k_agents, k_dirs, k_obs = jax.random.split(key, 4) # splitting up the sources of randomness to avoid correlation...one source of randomness for agent spawn position; another for direction and observation

            agent_pos = jax.random.permutation(k_agents, self.SPAWNS_PLAYERS)[:num_agents] # we randomly asign the agent' starting positions
            agent_dirs = jax.random.randint( # randomly assign direction that agent faces ( up, down, left, right...that is why maxval = 4)
                k_dirs, shape=(num_agents,), minval=0, maxval=4, dtype=jnp.int16
            )
            agent_locs = jnp.stack( # combine agent position with agent direction 
                [agent_pos[:, 0], agent_pos[:, 1], agent_dirs], axis=-1
            )

            # initialise the grid according to the ASCII Map, and with random positions and directions (agent_locs) of agent:
            grid = jnp.zeros((self.GRID_SIZE_ROW, self.GRID_SIZE_COL), dtype=jnp.int16)
            grid = grid.at[self.SPAWNS_WALL[:,0], self.SPAWNS_WALL[:,1]].set(jnp.int16(Items.wall))
            grid = grid.at[self.RIVER[:,0], self.RIVER[:, 1]].set(jnp.int16(Items.river))
            grid = grid.at[agent_locs[:,0], agent_locs[:, 1]].set(self._agents)

            # we obtain some initial observation for the agents: 
            obs_noise = jax.random.normal(k_obs, shape=(num_agents,))* self.sigma_noise
            river_obs_init = jnp.clip(1.0+obs_noise, 0.0, 1.0)

            return State(
                agent_locs=agent_locs,
                river_level=jnp.float32(1.0),
                energy=jnp.full((num_agents,), 0.5, dtype=jnp.float32), # note that we initialise agents' energy level at 0.5...not starving, not full of energy either. 
                reputations=jnp.zeros((num_agents,), dtype=jnp.float32),
                cumulative_harvest=jnp.zeros((num_agents,), dtype=jnp.float32),
                cumulative_invest=jnp.zeros((num_agents,),dtype=jnp.float32),
                num_steps_below_collapse=jnp.int32(0),
                river_obs=river_obs_init,
                grid=grid,
                inner_t=jnp.int32(0),
                outer_t=jnp.int32(0),
            )
        
        # Now we return both the observation and state for the initial state: 
        def reset(key: jnp.ndarray):
            state = _reset_state(key)
            obs = _get_obs(state)
            return obs, state
        
        if jit: # compiles much faster using JIT
            self.step_env = jax.jit(_step)
            self.reset = jax.jit(reset)
        else:
            self.step_env = _step
            self.reset = reset

    @property
    def name(self) -> str:
        return "ISP"
    
    @property
    def num_actions(self) -> int: 
        return 9 + self.num_agents
    
    def action_space(self, agent_id=None) -> Discrete:
        """Action space is 9 base actions + 1 PUNISH(j) per agent (turn left/right, move in 4 directions an stay, harvest and invest)"""
        return Discrete(9 + self.num_agents)
    
    def observation_space(self):
        """
        Observation shape per agent: 
        - one-hot grid channels: len(Items)-1 + num_agents (= 2 + num_agents: wall, river, each agent)
        - scalar channels: 2 (river_obs, energy)
        Total channels = 4 + num_agents 
        """
        num_classes = len(Items) - 1 + self.num_agents
        total_channels = num_classes + 2 # here we add 2 for the river_obs and energy
        shape = (
            (self.OBS_SIZE, self.OBS_SIZE, total_channels)
            if self.cnn
            else (self.OBS_SIZE**2*total_channels,)
        )
        return Box(low=0.0, high=1.0, shape=shape, dtype=jnp.float32), shape
    
    def render_tile(
            self,
            obj: int,
            agent_dir=None,
            highlight: bool=False,
            tile_size: int=32,
            subdivs: int=3,
    ) -> onp.ndarray:
        key = (obj, agent_dir, highlight, tile_size)
        if key in self.title_cache: # cache tile so repeated renders do not recompute. 
            return self.title_cache[key]
        
        img = onp.full(
            shape = (tile_size*subdivs, tile_size*subdivs, 3),
            fill_value=(190,170,120),
            dtype=onp.uint8,
        )

        if obj == Items.wall: 
            fill_coords(img, point_in_rect(0,1,0,1), (127.0,127.0,127.0)) #grey rectangle for wall
        elif obj == Items.river: 
            fill_coords(img, point_in_rect(0,1,0,1), (40.0, 80.0, 214.0)) # blue rectangle for river
        elif obj in self._agents:
            agent_color = self.PLAYER_COLOURS[int(obj) - len(Items)]
            fill_coords(img, point_in_circle(0.5,0.5,0.31), agent_color)
            if agent_dir is not None:
                tri_fn = point_in_triangle(
                    (0.12, 0.19), (0.87, 0.50), (0.12, 0.81) # white direction triangle for the direction that agent is pointing 
                )
                tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi * agent_dir)
                fill_coords(img, tri_fn, (255.0, 255.0, 255.0))

        if highlight:
            highlight_img(img)

        img = downsample(img, subdivs)
        self.title_cache[key] = img
        return img

# TODO: last_claims, reputations (once communication logic is addedd), render method 

            


