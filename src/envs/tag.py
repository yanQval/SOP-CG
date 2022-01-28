from smac.env.multiagentenv import MultiAgentEnv as MultiAgentEnv_
from envs.MPE.make_env import make_env
from envs.MPE.multiagent.environment import MultiAgentEnv
import envs.MPE.multiagent.predator_prey as predator_prey

import numpy as np
import torch as th
import time


class TagEnv(MultiAgentEnv_):
    def __init__(
            self,
            n_good_agents=1,
            n_adversaries=3,
            n_landmarks=2,
            episode_limit=25,
            benchmark=False,
            reward_scale=0.1,
            object_scale=1,
            obs_range=100,
            seed=None
    ):
        scenario = predator_prey.Scenario()
        # create world
        world = scenario.make_world(n_good_agents, n_adversaries, n_landmarks, object_scale, obs_range)
        # create multiagent environment
        if benchmark:        
            self.env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.state, scenario.benchmark_data)
        else:
            self.env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.state)

        self.n_good_agents = n_good_agents
        self.n_adversaries = n_adversaries
        self.n_landmarks = n_landmarks
        self.n_actions = 5
        self.reward_scale = reward_scale
        self.object_scale = object_scale

        observation_space = self.env.observation_space
        self.obs_shape = []
        for space in observation_space:
            self.obs_shape.append(space.shape[0])
        self.episode_limit = episode_limit
        self._total_steps = 0
        self.battles_game = 0

        self.obs, self.state = self.env.reset()
        self._episode_steps = 0

    def step(self, adversaries_actions, good_agents_actions=None):
        """Returns reward, terminated, info."""
        self._total_steps += 1
        self._episode_steps += 1
        info = {}
        terminated = False

        # for i in range(self.n_adversaries):
        #     x = self.obs[i][-4]
        #     y = self.obs[i][-3]
        #     print(max(max(x, y), max(-x, -y)))
        #     if max(max(x, y), max(-x, -y)) == x:
        #         adversaries_actions[i] = 2
        #     if max(max(x, y), max(-x, -y)) == y:
        #         adversaries_actions[i] = 4
        #     if max(max(x, y), max(-x, -y)) == -x:
        #         adversaries_actions[i] = 1
        #     if max(max(x, y), max(-x, -y)) == -y:
        #         adversaries_actions[i] = 3
        # print(self.obs)
        # print(adversaries_actions)
        # self.env.render(mode=None)
        # time.sleep(0.1)

        if good_agents_actions is None:
            good_agents_actions = th.randint(low=0, high=self.n_actions, size=(self.n_good_agents,), device=adversaries_actions.device, dtype=adversaries_actions.dtype)
        actions = th.cat([adversaries_actions, good_agents_actions])

        self.obs, self.state, rewards, done, infos = self.env.step(actions)
        reward = rewards[0] * self.reward_scale
        good_agents_reward = np.sum(rewards[self.n_adversaries:]) * self.reward_scale

        if self._episode_steps >= self.episode_limit:
            terminated = True

        if terminated:
            self.battles_game += 1

        return reward, good_agents_reward, terminated, info

    def get_obs(self):
        """Returns all agent observations in a list."""
        return self.obs[:self.n_adversaries]

    def adv_get_obs(self):
        """Returns all agent observations in a list."""
        return self.obs[self.n_adversaries:]

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_shape[0]

    def adv_get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_shape[-1]

    def get_state(self):
        """Returns the global state."""
        return self.state

    def get_state_size(self):
        """Returns the size of the global state."""
        return (self.n_adversaries + self.n_good_agents) * 4 + self.n_landmarks * 2

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1] * self.n_actions for i in range(self.n_adversaries)]

    def adv_get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1] * self.n_actions for i in range(self.n_good_agents)]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def adv_get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def reset(self):
        """Returns initial observations and states."""
        self.obs, self.state = self.env.reset()
        self._episode_steps = 0

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
                    "n_agents": self.n_adversaries,
                    "adv_obs_shape": self.adv_get_obs_size(),
                    "adv_n_actions": self.adv_get_total_actions(),
                    "adv_n_agents": self.n_good_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game
        }
        return stats
