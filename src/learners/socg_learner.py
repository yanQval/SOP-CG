import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop
import utils.constructor as constructor


class SocgLearner:
    
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        if self.args.construction == 'matching':
            self.solver = constructor.MatchingSolver(args)
        elif self.args.construction == 'tree':
            self.solver = constructor.TreeSolver(args)
        else:
            raise Exception("unimplemented method")

        self.constructor = constructor.Constructor(args)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        graphs = batch["graphs"]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        chosen_action_qvals = []
        target_max_qvals = []
        self.mac.init_hidden(batch.batch_size)
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            # Calculate target values

            f, g, target_graphs_ = self.mac.forward(batch, t=t, select_graph=True)
            agent_outs = self.constructor.compute_outputs(f, g, avail_actions[:, t], target_graphs_)
            agent_outs_detach = agent_outs.detach()
            agent_outs_detach[avail_actions[:, t] == 0] = -9999999
            cur_max_actions = agent_outs_detach.max(dim=-1, keepdim=True)[1]

            target_f, target_g = self.target_mac.forward(batch, t=t)
            if self.args.double_q_on_graph:
                target_max_qvals.append(self.constructor.compute_values_given_actions(target_f, target_g, cur_max_actions, target_graphs_))
            else:
                target_graphs = self.solver.solve_given_actions(target_f, target_g, cur_max_actions, device=self.args.device)
                target_max_qvals.append(self.constructor.compute_values_given_actions(target_f, target_g, cur_max_actions, target_graphs))

            # Calculate estimated Q-Values for the current actions
            if t < batch.max_seq_length - 1:
                # graphs_used = graphs[:, t]
                graphs_used = self.solver.solve_given_actions(f, g, actions[:, t], device=self.args.device)
                graphs_used = self.solver.graph_epsilon_greedy(graphs_used, self.args.graph_epsilon)
                mac_values = self.constructor.compute_values_given_actions(f, g, actions[:, t], graphs_used)
                chosen_action_qvals.append(mac_values)

                if self.args.communicate:
                    self.mac.communicate(graphs_used)
                    self.target_mac.communicate(graphs_used)

        chosen_action_qvals = th.stack(chosen_action_qvals, dim=1).unsqueeze(-1)
        target_max_qvals = th.stack(target_max_qvals[1:], dim=1).unsqueeze(-1)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # # Normal L2 loss, take mean over actual data
        # loss = (masked_td_error ** 2).sum() / mask.sum()
        # L2 loss with asymmetric objective
        loss = ((masked_td_error * (masked_td_error >= 0)) ** 2).sum() / mask.sum() \
               + self.args.asym_alpha * ((masked_td_error * (masked_td_error < 0)) ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
