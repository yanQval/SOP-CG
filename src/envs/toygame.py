

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging
import random


class ToygameEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(
            self,
            n_groups=4,
            n_agents_in_group=3,
            episode_limit=1,
            seed=None
    ):
        # Map arguments
        self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)
        self.n_groups = n_groups
        self.n_agents_in_group = n_agents_in_group
        self.n_agents = self.n_groups * self.n_agents_in_group
        self.n_actions = 2

        # Other
        self._seed = seed

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.episode_limit = episode_limit
        
        self.group_id = []
        for i in range(self.n_agents):
            self.group_id.append(0)
    
        self.random_allo()
    
    def random_allo(self):
        self.p = np.random.permutation([i for i in range(self.n_agents)])
        cnt = 0
        self.label = []
        for i in range(self.n_agents):
            self.label.append(0)
        pos_g = np.random.randint(0, self.n_groups)
        for i in range(0, self.n_agents, self.n_agents_in_group):
            for j in range(i, i + self.n_agents_in_group):
                self.group_id[self.p[j]] = cnt
            if i == pos_g * self.n_agents_in_group:
                for j in range(i, i + self.n_agents_in_group):
                    self.label[self.p[j]] = 1
            else:
                x = np.random.randint(i, i + self.n_agents_in_group)
                for j in range(i, i + self.n_agents_in_group):
                    if j == x:
                        self.label[self.p[j]] = 0
                    else:
                        self.label[self.p[j]] = 1
            cnt += 1

    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        count = 0
        reward = 0
        #print(self.p)
        for i in range(0, self.n_agents, self.n_agents_in_group):
            cnt = 0
            for j in range(i, i + self.n_agents_in_group):
                if actions[self.p[j]] == 1:
                    cnt += 1
            if cnt == self.n_agents_in_group:
                reward += 1
            else:
                reward -= cnt * 0.5
        
        terminated = False
        info['battle_won'] = False

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

            if count == self.n_groups:
                info['battle_won'] = True
                self.battles_won += 1

        # return reward, terminated, info, self.group_id
        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        obs = np.zeros(self.n_groups)
        obs[self.group_id[agent_id]] = 1
        return np.concatenate((obs, [self.label[agent_id]]))

    def get_obs_size(self):
        """Returns the size of the observation."""
        return 1 + self.n_groups

    def get_state(self):
        """Returns the global state."""
        return np.zeros(1)

    def get_state_size(self):
        """Returns the size of the global state."""
        return 1

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [1, self.label[agent_id]]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0
        
        self.random_allo()
        
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game
        }
        return stats
