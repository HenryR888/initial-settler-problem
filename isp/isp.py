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
    last_comms: jnp.ndarray # (n,4)...this is our last_comm tuple per agent, which is the [claim_level, accuse_target, charge, recommend]
    tile_richness: jnp.ndarray # (num_river_tiles, ) richness of each river tile, which is resampled each episode

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
            g = -0.05, # NoOp energy gain for agent not doing anything at that specific number step
            K_collapse_thresh = 0.1, # river health level collapse threshold (after k steps below threshold, episode transitions into terminal state)
            k_collapse_steps = 15, # number of steps below K before episode collapses (boosted to 15 to allow agents more time before river collapses during early training) (CHANGED FROM 5)
            sigma_noise = 0.05, # standard deviation for Normal distribution for noise
            #reward function hyperparams:
            w_f=1.0, # weight for delta energy
            w_h = 1.0, # weight for Indicator function for agent being below survival threshold...if agent is below threshold - huge penalty
            w_c = 1.0, # weight for indicator function for river health being below K collapse shreshold
            w_p=0.1, # weight for indicator function for agent being punished
            lambda_h = 0.1, # hunger threshold for energy level...if energy level drops below this, agent is penalised heavily with w_h
            #lambda_c=0.2, # river collapse hyperparam...we shall start with K as this value, but might provide interesting dynamics to have different K values with lambdda_c to start penalising agents earlier rather than later
            c_pun=0.08, # energy cost for agent to punish someone else
            c_rec=0.16, # energy cost for agent who is receiving punishment
            jit = True,
            obs_size=5, # each agent has fov of 5x5 grid within tiles to rely on comms
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
                ],
            # reputation and audit hyperparams:
            p_err=0.1, # error of faulty communication i.e. audit signal is flipped
            qmax=1.5, # greedy (harvest-invest ratio) threshold, beyond which agent is considered to freeload
            freeload_thresh=0.05, # invest threshold below which we shall consider the agent is freeloading...need both because ratio is not enough (i.e. free loading also should be considered if agent harvests same amount as investing)
            lie_tolerance=1, # one claim_bin level off is allowed (could have been erroneous due to error)...but more than 1 is considered a strategic lie
            delta_rep=0.05, # change in reputation score...!NOTE perhaps we adjust this to a function-based reputation
            eps_greed=1e-3, # epsilon added to harvest-invest (greed ratio) to prevent division by zero
            K_claim_bins=5, # [very_low, low, medium, high, very high] claim for river level
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
        self.p_err = p_err
        self.qmax = qmax
        self.freeload_thresh = freeload_thresh
        self.lie_tolerance = lie_tolerance
        self.delta_rep = delta_rep
        self.eps_greed = eps_greed
        self.K_claim_bins = K_claim_bins
        self.COMM_NORM = jnp.array([K_claim_bins-1, num_agents, 3, 3], dtype=jnp.float32) # normalise the communication component to [0,1] before putting it into obs
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

            
            def add_scalar_channels(grid, agent_idx):
                # get the tile_richness for each tile here
                def get_tile_richness(loc):
                    matches = jnp.all(self.RIVER == loc[:2], axis=-1)                                                                                                                                
                    return jnp.where(jnp.any(matches), state.tile_richness[jnp.argmax(matches)], 1.0) # we return 1.0 for neural if the agent is not standing on a river tile. 
                # river channel for each agent: 
                river_ch = jnp.full(
                    (self.OBS_SIZE, self.OBS_SIZE, 1),
                    state.river_obs[agent_idx], dtype=jnp.float32,
                )
                # energy channel for each agent: 
                energy_ch = jnp.full(
                    (self.OBS_SIZE, self.OBS_SIZE, 1),
                    state.energy[agent_idx], dtype=jnp.float32,
                )
                # reputation channel for each agent's reputation: 
                rep_chs = jnp.broadcast_to(
                    state.reputations.reshape(1,1,num_agents),
                    (self.OBS_SIZE, self.OBS_SIZE, num_agents),
                )
                # normalise the agents; last comm tuples to 1 for better optimisation: 
                normed_comms = state.last_comms/self.COMM_NORM # (n,4)
                # here we have an inbox which is global and made up of all the 4 parts of the comm vector that the agents will be able to perceive. 
                inbox = jnp.broadcast_to(
                    normed_comms.reshape(1,1,4*num_agents),
                    (self.OBS_SIZE, self.OBS_SIZE, 4*num_agents),
                )
                # richness channel for the richness of the tile: 
                richness_ch = jnp.full(
                    (self.OBS_SIZE, self.OBS_SIZE, 1),
                    get_tile_richness(state.agent_locs[agent_idx]), dtype=jnp.float32,
                )
                return jnp.concatenate([grid, river_ch, energy_ch, rep_chs, richness_ch, inbox], axis=-1)
            
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
            # Moreover, we do not want collision key to be correlated with river noise or obs noise, otherwise agents could use collision dynamics to infer state of river/observation of other agents. Also add audit key for communcation audit signal error
            key, k_river_noise, k_obs_noise, k_collision, k_audit_err, k_spawn = jax.random.split(key, 6)

            actions = jnp.array(actions) # shape (num_agents, 5)
            env_actions = actions[:,0] # move/harvest/invest/punish
            claim_levels = actions[:,1] # river claim (0,4)
            accuse_targets = actions[:,2] # which agent we accuse (0=no one, 1...n= agent index+1)...agent cannot accuse himself
            charges = actions[:,3] # charge type (0=none, 1=greedy, 2=freeload, 3=lie)
            recommends = actions[:,4] # recommendation (0=none, 1=harvest-less, 2=harvest-more, 3=invest-more)

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
            )(state.agent_locs, env_actions).squeeze()

            agent_move = ( # boolean mask to update agents' locations in the next step, given that they chose a movement action
                (env_actions == Actions.up) | (env_actions == Actions.down) | 
                (env_actions == Actions.left) | (env_actions == Actions.right)
            )
            all_new_locs = jax.vmap( # move all the agents accordingly to their new respective locations
                lambda m, n, a: jnp.where(m,n+self.STEP_MOVE[a], n)
            )(agent_move, all_new_locs, env_actions)

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
            harvesting = (env_actions == Actions.harvest) & on_river
            investing = (env_actions == Actions.invest) & on_river
            noop = (env_actions == Actions.stay)
            punish_target = jnp.clip(env_actions-9, 0, num_agents-1)
            punishing = (env_actions>=9) & (punish_target !=jnp.arange(num_agents)) # prevent agent from punishing itself. 
            

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
            
            energy_new = energy_old + energy_delta # here we removed the clipped energy, and leave it uncapped

            # River Dynamics Update: recall R_{t+1} = clip(R_t - D_t + I_t + eps_t, 0, 1)
            D_t = jnp.sum(jnp.where(harvesting, self.gamma_h, 0.0)) # damage to be subtracted from river health from agents harvesting
            I_t = jnp.sum(jnp.where(investing, self.gamma_v, 0.0)) # health boost to be added back to river from agents investing 
            eps_t = jax.random.normal(k_river_noise)* self.sigma_noise # eps_t ~ N(0, sigma^2)

            R_t = state.river_level
            R_new = jnp.clip( # update river level according to behaviourial-dependent ecology process above
                R_t - D_t + I_t + eps_t, 
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

            # Social Audit & Reputation Update: 

            # True metrics (computed from the cumulative counts):
            greed_true = (
                (new_cumulative_harvest+self.eps_greed)/
                (new_cumulative_invest+self.eps_greed)
            ) > self.qmax

            avg_invest_per_step = new_cumulative_invest/(state.inner_t+1.0)
            freeload_true = avg_invest_per_step < self.freeload_thresh

            # we have set 5 bins here for the R_t river health: [VERY LOW, LOW, MEDIUM, HIGH, VERY HIGH].
            # We take the actual R_t (R_new latest update of river) and multiply by 5. Thus we have the following corresponding mapping:
            # 0=[0,0.2), 1=[0.2,0.4), 2=[0.4,0.6), 3=[0.6,0.8), 4=[0.8,1.0]
            realised_bin = jnp.clip(
                jnp.floor(R_new*self.K_claim_bins).astype(jnp.int32), 0, self.K_claim_bins -1
            )
            # here we check to see if the agent's previous claim was off compared to the realised river level by more than the lie_tolerance (default is set to 1). I.e. agent claimed river was low when it was actually very low is not considered strategic lie. But agent claiming river is high when agent is low, is off by 2 and thus is considered a lie.
            lie_true = (
                jnp.abs(state.last_comms[:,0].astype(jnp.int32) - realised_bin) > self.lie_tolerance
            )

            # Accusation is valid only when both the target and charge entries are non-zero: 
            is_accusing = (accuse_targets>0)&(charges>0)
            accused_idx = jnp.clip(accuse_targets-1, 0, num_agents-1) # here we are 0 indexing 
            is_accusing = is_accusing & (accused_idx != jnp.arange(num_agents)) # this resolves the self-accusation problem (agent cannot accuse himself)

            # select the true metric relevant to each accuser's charge: 
            true_metric_for_accused = jnp.where(
                charges == 1, greed_true[accused_idx], # check to see if it is a greed charge, is agent actually greedy? 
                jnp.where(charges==2, freeload_true[accused_idx], # if it is a freeload charge, check if that agent actually freeloaded
                jnp.where(charges==3, lie_true[accused_idx], # if it is a lie charge, check if that agent actually lied
                          jnp.zeros(num_agents, dtype=jnp.bool_))) # otherwise nothing
            )

            # Noisy audit signal: y = true_metric XOR Bernoulli(p_err) as per docs: 
            audit_err = jax.random.bernoulli(k_audit_err, self.p_err, shape=(num_agents,))
            audit_signal = jnp.logical_xor(true_metric_for_accused, audit_err)

            # Reputation dynamics: 
            # for now if the accused agent is found guilty then the target's reputation drops
            # and if the accused agent is found innocent, then the accuser's reputation drops (this is a false accusation)
            rep_delta = jnp.zeros((num_agents,), dtype=jnp.float32)

            #!NOTE: we might want to consider adding a positive reputation delta for accusing correctly after first experiment run. The reason I opt for not doing it initially is because I want to mimic how socially one is not rewarded directly for making a correct accusation. But rather indirectly, by accusing and punishing correctly, the agent is punishing freeloaders, liers and greedy harvesters from the system which maintains the longevity and stability of the environmnet

            def guilty_delta(j):
                guilty = is_accusing & audit_signal & (accused_idx == j)
                return jnp.sum(jnp.where(guilty, -self.delta_rep, 0.0))
            
            rep_delta = rep_delta + jax.vmap(guilty_delta)(jnp.arange(num_agents))
            rep_delta = rep_delta - jnp.where(is_accusing & ~audit_signal, self.delta_rep, 0.0) # reduce the accuser's reputation if false accusation

            new_reputations = jnp.clip(state.reputations + rep_delta, 0.0, 1.0)

            # RESPAWN Conditions, recall that for ISP 5.0, we want the agents to respawn when they are punished, if their energy hits 0 and if the agent is found guilty. 
            received_punishment = jax.vmap(
                lambda j: jnp.any(punishing & (punish_target == j))
            )(jnp.arange(num_agents))

            found_guilty = jax.vmap(
                lambda j: jnp.any(is_accusing & audit_signal & (accused_idx == j))
            )(jnp.arange(num_agents))

            starved = energy_new <=0

            respawn = received_punishment | found_guilty | starved

            # now we choose a random spawn tile for each agent (only if respawn is TRUE): 
            random_spawn_idxs = jax.random.randint(k_spawn, shape=(num_agents,), minval=0, maxval=len(self.SPAWNS_PLAYERS))
            random_spawn_locs = self.SPAWNS_PLAYERS[random_spawn_idxs]
            spawn_locs = jnp.concatenate([random_spawn_locs, jnp.zeros((num_agents, 1), dtype=jnp.int16)], axis=-1)

            new_locs = jnp.where(respawn[:, None], spawn_locs, new_locs)
            energy_new = jnp.where(respawn, 0.5, energy_new)


            # Update last_comms for this step: 
            new_last_comms = jnp.stack(
                [claim_levels, accuse_targets, charges, recommends], axis=-1
            ).astype(jnp.float32)

            # here we get the richness of each river tile:
            def get_tile_richness(loc): 
                matches = jnp.all(self.RIVER == loc[:2], axis=-1) # (num_river_tiles,)
                return jnp.where(jnp.any(matches), state.tile_richness[jnp.argmax(matches)], 1.0)
            agent_tile_richness = jax.vmap(get_tile_richness)(new_locs) #(num_agents,)

            # Reward function Update: Recall reward is: u_{t,i} = w_f*delta_e - w_h.I[e<=lambda_h] - w_c*I[r<=k] - w_p*[punish]
            #delta_e = energy_new - energy_old
            #hunger_penalty = jnp.where(energy_new <= self.lambda_h, self.w_h, 0.0)
            #collapse_penalty = jnp.where(R_new<=self.K_collapse_thresh, self.w_c, 0.0)
            #punish_penalty = jnp.where(punishing, self.w_p, 0.0)
            # rewards = (self.w_f*delta_e) - hunger_penalty - collapse_penalty - punish_penalty # reward function
            rewards = harvesting.astype(jnp.float32) * R_new * agent_tile_richness # update reward to depend on the level of the river and tile richness in order to provoke communication

            # update the grid with river tiles and place agents on their new respective tiles: 
            new_grid = state.grid.at[self.RIVER[:, 0], self.RIVER[:, 1]].set(jnp.int16(Items.river))
            new_grid = new_grid.at[new_locs[:, 0], new_locs[:, 1]].set(self._agents)

            state_nxt = State( # build out the next state at time t+1
                agent_locs=new_locs,
                river_level=R_new,
                energy=energy_new,
                reputations=new_reputations, # we haven't changed this yet...once adding audit logic then we will change reputations vector
                cumulative_harvest=new_cumulative_harvest,
                cumulative_invest=new_cumulative_invest,
                num_steps_below_collapse=num_steps_below,
                river_obs=river_obs_new,
                last_comms=new_last_comms,
                grid=new_grid,
                inner_t = state.inner_t+1,
                outer_t=state.outer_t,
                tile_richness=state.tile_richness,
            )

            # episode reset logic:
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            collapse_done = num_steps_below>=self.k_collapse_steps # reset if the number of steps that river is below collapse threshold reaches number of collapse steps
            reset_inner = (inner_t == num_inner_steps) | collapse_done # also reset if reached total number of time steps specified within the episode

            state_re = _reset_state(key)
            state_re = state_re.replace(
                outer_t = jnp.where(outer_t+1 >= num_outer_steps, jnp.int32(0), outer_t +1 )
            )

            state = jax.tree_map( # if the episode is terminated, then reset, otherwise, go to the next state
                lambda x, y: jnp.where(reset_inner, x, y),
                state_re,
                state_nxt
            )

            # check to see that training is done, if so then give done flag
            reset_outer = reset_inner & (outer_t + 1 >= num_outer_steps)
            done = {f'{a}': reset_outer for a in self.agents}
            done["__all__"] = reset_outer

            obs = _get_obs(state)
            rewards = jnp.where(reset_inner, jnp.zeros_like(rewards), rewards) # reset rewards after episode complete

            info = { # return some diagnostics which we can go through for our own reference...this is not for the agents' training
                "harvest": harvesting.astype(jnp.float32),
                "invest": investing.astype(jnp.float32),
                "punish": punishing.astype(jnp.float32),
                "river_level": R_new,
                "energy": energy_new,
                "collapse": collapse_done,
                "reputations": new_reputations,
            }

            return obs, state, rewards, done, info
        
        # reset function which also initialises the state of the episode:
        def _reset_state(
                key: jnp.ndarray
        ) -> State:
            key, k_agents, k_dirs, k_obs, k_rich = jax.random.split(key, 5) # splitting up the sources of randomness to avoid correlation...one source of randomness for agent spawn position; another for direction and observation, and another for the tile richness

            agent_pos = jax.random.permutation(k_agents, self.SPAWNS_PLAYERS)[:num_agents] # we randomly asign the agent' starting positions
            agent_dirs = jax.random.randint( # randomly assign direction that agent faces ( up, down, left, right...that is why maxval = 4)
                k_dirs, shape=(num_agents,), minval=0, maxval=4, dtype=jnp.int16
            )
            agent_locs = jnp.stack( # combine agent position with agent direction 
                [agent_pos[:, 0].astype(jnp.int16), agent_pos[:, 1].astype(jnp.int16), agent_dirs], axis=-1
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
                reputations=jnp.full((num_agents,), 0.5, dtype=jnp.float32), # initialise agents' reputation at 0.5 each (neutral) so that reputation can move in either direction. 
                cumulative_harvest=jnp.zeros((num_agents,), dtype=jnp.float32),
                cumulative_invest=jnp.zeros((num_agents,),dtype=jnp.float32),
                num_steps_below_collapse=jnp.int32(0),
                river_obs=river_obs_init,
                grid=grid,
                inner_t=jnp.int32(0),
                outer_t=jnp.int32(0),
                last_comms=jnp.zeros((num_agents, 4), dtype=jnp.float32),
                tile_richness=jax.random.uniform(k_rich, shape=(len(self.RIVER),), minval=0.5, maxval=1.5)
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
    
    def action_space(self, agent_id=None):
        return Tuple([
            Discrete(9+self.num_agents), # environment actions
            Discrete(self.K_claim_bins), # claim level of river health
            Discrete(self.num_agents+1), # accuse target (0=nobody)
            Discrete(4), # charge actions
            Discrete(4), # recommend actions
        ])
        
    
    def observation_space(self):
        """
        Observation shape per agent: 
        - one-hot grid channels: len(Items)-1 + num_agents (= 2 + num_agents: wall, river, each agent)
        - scalar channels:  river_obs, energy, reputations (same as num_agents) and inbox which is (4 comm actions*num agents)
        """
        num_classes = len(Items) - 1 + self.num_agents
        total_channels = num_classes + 3 + self.num_agents + 4*self.num_agents # channels are representative of river_obs, energy, reputations and inbox and richness channel
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

            


