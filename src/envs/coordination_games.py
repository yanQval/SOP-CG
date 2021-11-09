

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv

import numpy as np
import random


class MatchingGameEnv(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(
            self,
            n_agents=6,
            n_actions=3,
            episode_limit=1,
            reward=1,
            seed=None
    ):
        self._seed = seed
        if seed == None:
            self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)

        # Map arguments
        self.n_agents = n_agents

        # Actions
        self.n_actions = n_actions
        self.reward = reward

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.episode_limit = episode_limit

        self.random_matching()

    def random_matching(self):
        self.edges = []
        p = [i for i in range(self.n_agents)]
        for i in range(0, self.n_agents, 2):
            self.edges.append((p[i], p[i + 1]))
        self.obs = np.zeros((self.n_agents, self.n_agents))
        for e in self.edges:
            self.obs[e[0], e[1]] = 1
            self.obs[e[1], e[0]] = 1

    # def step(self, actions, g, graph, test_mode):
    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        reward = 0
        cnt = 0

        for i, e in enumerate(self.edges):
            if (actions[e[0]] + i) % self.n_actions == actions[e[1]]:
                reward += self.reward
                cnt += 1
            else:
                reward -= self.reward

        # g = g.cpu().detach().numpy()
        # if self._total_steps % 300 == 0 and test_mode == True:
        #     print(self.edges)
        #     print(actions)
        #     print(reward)
        #     for i in range(self.n_agents):
        #         for j in range(i + 1, self.n_agents):
        #             print(i, j, g[i, j])
        #     print(graph)

        terminated = False
        info['battle_won'] = False

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

            if cnt == len(self.edges):
                info['battle_won'] = True
                self.battles_won += 1

        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.obs[agent_id]

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.n_agents

    def get_state(self):
        """Returns the global state."""
        return []

    def get_state_size(self):
        """Returns the size of the global state."""
        return 0

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [1] * self.n_actions

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self._episode_steps = 0
        self.random_matching()

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
