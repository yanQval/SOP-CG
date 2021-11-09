from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from .episode_runner import EpisodeRunner
import utils.matching as matching


class CGEpisodeRunner(EpisodeRunner):

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.cnt = 0
        self.estimated_Qs = []

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        episode_reward = 0
        gamma_now = 1
        gamma = 0.99
        first = False
        estimated_Q = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            graphs, actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            f, g = self.mac.forward(self.batch, t=self.t, test_mode=test_mode, select_graph=False)
            if self.args.communicate:
                self.mac.communicate(graphs)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            episode_reward += reward * gamma_now
            gamma_now *= gamma
            if not first:
                #print(f.shape)
                #print(g.shape)
                #print(actions.unsqueeze(2))
                #print(graphs)
                estimated_Q = matching.compute_values_given_actions(f, g, actions.unsqueeze(2), graphs)
                self.estimated_Qs.append(estimated_Q.item())
                first = True
                #print(estimated_Q)


            post_transition_data = {
                "graphs": graphs,
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        graphs, actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"graphs": graphs, "actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_reward)

        # if test_mode and (len(self.test_returns) == self.args.test_nepisode):
        #     self._log(cur_returns, cur_stats, log_prefix)
        # elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
        #     self._log(cur_returns, cur_stats, log_prefix)
        #     if hasattr(self.mac.action_selector, "epsilon"):
        #         self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
        #     self.log_train_stats_t = self.t_env

        self.cnt += 1
        if self.cnt == self.args.test_nepisode:
            print('mean over', self.args.test_nepisode,'runs')
            print('estimated_Q')
            print(np.mean(self.estimated_Qs))
            print('real_rewards')
            print(np.mean(cur_returns))

        return self.batch