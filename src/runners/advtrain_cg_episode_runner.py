from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from .episode_runner import EpisodeRunner


class AdvCGEpisodeRunner(EpisodeRunner):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.adv_train_returns = []
        self.adv_test_returns = []

    def setup(self, scheme, groups, preprocess, mac, adv_scheme, adv_groups, adv_preprocess, adv_mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.adv_new_batch = partial(EpisodeBatch, adv_scheme, adv_groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.adv_mac = adv_mac

    def reset(self):
        self.batch = self.new_batch()
        self.adv_batch = self.adv_new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        adv_episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.adv_mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            adv_pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.adv_get_avail_actions()],
                "obs": [self.env.adv_get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)
            self.adv_batch.update(adv_pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            graphs, actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            if self.args.communicate:
                self.mac.communicate(graphs)
            adv_actions = self.adv_mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, adv_reward, terminated, env_info = self.env.step(actions[0], adv_actions[0])
            episode_return += reward
            adv_episode_return += adv_reward

            post_transition_data = {
                "graphs": graphs,
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            adv_post_transition_data = {
                "actions": adv_actions,
                "reward": [(adv_reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)
            self.adv_batch.update(adv_post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        adv_last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.adv_get_avail_actions()],
            "obs": [self.env.adv_get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        self.adv_batch.update(adv_last_data, ts=self.t)


        # Select actions in the last stored state
        graphs, actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"graphs": graphs, "actions": actions}, ts=self.t)
        adv_actions = self.adv_mac.select_actions(self.adv_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.adv_batch.update({"actions": adv_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        adv_cur_returns = self.adv_test_returns if test_mode else self.adv_train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        adv_cur_returns.append(adv_episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
            self._log(adv_cur_returns, cur_stats, log_prefix + 'adv_')
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            self._log(adv_cur_returns, cur_stats, log_prefix + 'adv_')
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, self.adv_batch