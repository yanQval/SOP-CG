
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv

import numpy as np
import random


class Bearer2Env(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """
    def __init__(
            self,
            n_agents=7,
            agent_distribution=[1, 2, 4],
            agent_capacity=[6, 3, 1],
            episode_limit=50,
            map_size=11,
            map_division=[3, 7],
            deliver_reward=1,
            pick_reward=0.1,
            deliver_cost=0,
            sight_range=4,
            gen_rate=1,
            gen_division=[0, 1, 7],
            seed=None
    ):
        self._seed = seed
        if seed == None:
            self._seed = random.randint(0, 9999)
        np.random.seed(self._seed)

        # Map arguments
        self.n_agents = n_agents
        self.map_size = map_size
        self.agent_distribution = agent_distribution
        self.agent_capacity = agent_capacity
        self.map_division = map_division
        self.sight_range = sight_range
        self.gen_rate = gen_rate
        self.gen_division = gen_division
        self.n_roles = len(agent_distribution)

        # Actions
        self.n_actions = 6
        self.deliver_reward = deliver_reward
        self.pick_reward = pick_reward
        self.deliver_cost = deliver_cost

        # Statistics
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self.battles_won = 0
        self.battles_game = 0
        self.episode_limit = episode_limit
        self.delivered_letters = 0
        self.total_reward = 0
        self.neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Configuration initialization
        self.agent_pos = np.zeros((self.n_agents, 2)).astype(int)
        self.agent_role = np.zeros(self.n_agents).astype(int)
        self.agent_has = np.zeros(self.n_agents)
        self.map = np.zeros((self.map_size, self.map_size))
        self.map_agent = [[[] for j in range(self.map_size)] for i in range(self.map_size)]

        tot = 0
        for (id, n) in enumerate(self.agent_distribution):
            for i in range(n):
                x, y = -1, -1
                while self.fit(x, y, id) is False:
                    x = np.random.randint(0, self.map_size)
                    y = np.random.randint(0, self.map_size)
                self.agent_pos[tot, 0] = x
                self.agent_pos[tot, 1] = y
                self.agent_role[tot] = id
                self.map_agent[x][y].append(tot)
                tot += 1

        self.gen_pos = []
        # for i in range(self.map_size):
        #    self.gen_pos.append((0, i))
        #    if i > 0:
        #        self.gen_pos.append((i, 0))
        #    self.gen_pos.append((self.map_size - 1, i))
        #    if i < self.map_size - 1:
        #        self.gen_pos.append((i, self.map_size - 1))
        for i in range(self.map_size):
            for j in range(self.map_size):
                for role in range(self.n_roles):
                    if self.fit(i, j, role):
                        for _ in range(self.gen_division[role]):
                            self.gen_pos.append((i, j))
                        break

        self.map_eye = np.eye(self.map_size)



    def step(self, actions):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}

        reward = 0
        info['battle_won'] = False

        # print(actions)

        # print('step', self._episode_steps)

        # print(self.map)
        
        # for x in range(self.map_size):
        #    for y in range(self.map_size):
        #        print(len(self.map_agent[x][y]), end=' ')
        #    print("")
        
        # for id, action in enumerate(actions):
        #    print(id, action)
        #    print(self.agent_pos[id])
        #    print(self.agent_has[id])

        if self.gen_rate >= 1.0:
            for _ in range(self.gen_rate):
                id = np.random.randint(0, len(self.gen_pos))
                self.map[self.gen_pos[id]] += 1
        else:
            if np.random.rand(1) < self.gen_rate:
                id = np.random.randint(0, len(self.gen_pos))
                self.map[self.gen_pos[id]] += 1

        for id, action in enumerate(actions):
            if action == 5:
                for id_other in range(id):
                    if self.agent_pos[id_other, 0] == self.agent_pos[id, 0] and self.agent_pos[id_other, 1] == self.agent_pos[id, 1] and self.agent_role[id_other] < self.agent_role[id]:
                        #print(self.agent_has[id], self.agent_capacity[self.agent_role[id_other]],  self.agent_has[id_other])
                        t = min(self.agent_has[id], self.agent_capacity[self.agent_role[id_other]] - self.agent_has[id_other])
                        reward += self.deliver_reward[self.agent_role[id]] * t
                        self.agent_has[id_other] += t
                        self.agent_has[id] -= t
                # min = id
                # for id_other in self.map_agent[self.agent_pos[id, 0]][self.agent_pos[id, 1]]:
                #    if id_other < min:
                #        min = id_other
                # if self.agent_role[min] < self.agent_role[id]:
                #    t = min(self.agent_has[id])
                #    reward += self.deliver_reward[self.agent_role[id]] * self.agent_has[id]
                #    self.agent_has[min] += self.agent_has[id]
                #    self.agent_has[id] = 0

        for id in range(self.n_agents):
            x, y = self.agent_pos[id]
            #print('111', self.map[x, y], self.agent_capacity[self.agent_role[id]] - self.agent_has[id])
            t = min(self.map[x, y], self.agent_capacity[self.agent_role[id]] - self.agent_has[id])
            self.agent_has[id] += t
            reward += t * self.pick_reward
            self.map[x, y] -= t
            # min = self.n_agents
            # for id in self.map_agent[x][y]:
            #    if id < min:
            #        min = id
            # self.agent_has[min] += self.map[x, y]
            # reward += self.map[x, y] * self.pick_reward
            # self.map[x, y] = 0

        for id, action in enumerate(actions):
            if 1 <= action <= 4:
                x, y = self.agent_pos[id]
                delta_x, delta_y = self.neighbors[action - 1]
                tx = x + delta_x
                ty = y + delta_y
                if self.fit(tx, ty, self.agent_role[id]):
                    self.map_agent[x][y].remove(id)
                    self.map_agent[tx][ty].append(id)
                    self.agent_pos[id] = (tx, ty)

        for id in self.map_agent[0][0]:
            reward += self.deliver_reward[self.agent_role[id]] * self.agent_has[id]
            self.delivered_letters += self.agent_has[id]
            self.agent_has[id] = 0

        terminated = False

        info['delivered_letters'] = self.delivered_letters

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self._episode_count += 1
            self.battles_game += 1

        # print('reward', self.total_reward)
        # if terminated:
        #    print('delivered', self.delivered_letters, info)
        #    print('---------')
        #
        # self.total_reward += reward

        return reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        grid_map = np.zeros((self.sight_range * 2 + 1, self.sight_range * 2 + 1))
        for x in range(self.map_size):
            for y in range(self.map_size):
                if self.map[x, y] > 0:
                    delta_x = x - self.agent_pos[agent_id, 0]
                    delta_y = y - self.agent_pos[agent_id, 1]
                    if abs(delta_x) <= self.sight_range and abs(delta_y) <= self.sight_range:
                        grid_map[delta_x + self.sight_range, delta_y + self.sight_range] = self.map[x, y]
        agents = np.zeros((self.n_agents, self.n_roles + 4))
        for id, (x, y) in enumerate(self.agent_pos):
            delta_x = x - self.agent_pos[agent_id, 0]
            delta_y = y - self.agent_pos[agent_id, 1]
            if abs(delta_x) <= self.sight_range and abs(delta_y) <= self.sight_range:
                agents[id, 0] = delta_x
                agents[id, 1] = delta_y
                agents[id, 2] = self.agent_has[id]
                agents[id, 3] = self.agent_capacity[self.agent_role[id]] - self.agent_has[id]
                agents[id, 4 + self.agent_role[id]] = 1
            else:
                agents[id, 0] = -1
                agents[id, 1] = -1
                agents[id, 2] = -1
                agents[id, 3] = -1
        # print(agent_id)
        # print(grid_map)
        role_onehot = np.zeros(self.n_roles)
        role_onehot[self.agent_role[agent_id]] = 1
        return np.concatenate([grid_map.flatten(), agents.flatten(), [self.agent_pos[agent_id, 0], self.agent_pos[agent_id,1 ]], role_onehot, [self.agent_has[agent_id]], [self.agent_capacity[self.agent_role[agent_id]] - self.agent_has[agent_id]]])




    def get_obs_size(self):
        """Returns the size of the observation."""
        return (self.sight_range * 2 + 1) ** 2 + self.n_agents * \
                    (4 + self.n_roles) + 2 + self.n_roles + 2

    def get_state(self):
        """Returns the global state."""
        return np.concatenate(self.get_obs())

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.n_agents * self.get_obs_size()

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
        # Statistics
        self._episode_steps = 0
        self.delivered_letters = 0
        self.total_reward = 0

        # Configuration initialization
        self.agent_pos = np.zeros((self.n_agents, 2)).astype(int)
        self.agent_role = np.zeros(self.n_agents).astype(int)
        self.agent_has = np.zeros(self.n_agents)
        self.map = np.zeros((self.map_size, self.map_size))
        self.map_agent = [[[] for j in range(self.map_size)] for i in range(self.map_size)]

        tot = 0
        for (id, x) in enumerate(self.agent_distribution):
            for i in range(x):
                x, y = -1, -1
                while self.fit(x, y, id) is False:
                    x = np.random.randint(0, self.map_size)
                    y = np.random.randint(0, self.map_size)
                self.agent_pos[tot, 0] = x
                self.agent_pos[tot, 1] = y
                self.agent_role[tot] = id
                self.map_agent[x][y].append(tot)
                tot += 1

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

    def fit(self, x, y, role):
        if role == 0:
            if 0 <= x < self.map_division[0] and 0 <= y < self.map_division[0]:
                return True
            else:
                return False
        if 0 <= x < self.map_division[role] and 0 <= y < self.map_division[role]:
            if x < self.map_division[role - 1] - 1 and y < self.map_division[role - 1] - 1:
                return False
            else:
                return True
        else:
            return False
