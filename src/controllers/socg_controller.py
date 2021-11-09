from .basic_controller import BasicMAC
from modules.agents import REGISTRY as agent_REGISTRY
import torch as th
import torch.nn as nn
import utils.constructor as constructor
import itertools


# This multi-agent controller shares parameters between agents
class SocgMAC(BasicMAC):

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions
        if args.construction == 'matching':
            self.solver = constructor.MatchingSolver(args)
        elif args.construction == 'tree':
            self.solver = constructor.TreeSolver(args)
        else:
            raise Exception("unimplemented method")

        self.comm_hidden_states = None
        self.input_shape = self.args.rnn_hidden_dim
        if self.args.communicate:
            self.comm_input_shape = args.comm_rnn_hidden_dim + self.n_agents
            self.comm_unit = agent_REGISTRY["comm_unit"](self.comm_input_shape, self.args)
            self.input_shape += self.args.comm_rnn_hidden_dim

        self.single_q = self._mlp(self.input_shape, args.single_q_hidden_dim, self.n_actions)
        self.pairwise_q = self._mlp(2 * self.input_shape, args.pairwise_q_hidden_dim, self.n_actions ** 2)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        f, g, graphs = self.forward(ep_batch, t_ep, test_mode=test_mode, select_graph=True)
        graphs, chosen_actions = self.action_selector.select_action(f[bs], g[bs], avail_actions[bs], graphs[bs], t_env, test_mode=test_mode)
        # if t_ep == 0:
        #     edges = []
        #     for i in range(self.n_agents):
        #         for j in range(i + 1, self.n_agents):
        #             if graphs[0, i, j] == 1:
        #                 edges.append((i, j))
        #             print((i, j), g[0, i, j])
        #     print('edges:', edges)
        #     print(chosen_actions)
        #     print(constructor.compute_values_given_actions(f, g, chosen_actions.unsqueeze(-1), graphs))
        return graphs, chosen_actions

    def forward(self, ep_batch, t, test_mode=False, select_graph=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        self.hidden_states = self.agent(agent_inputs, self.hidden_states).view(-1, self.n_agents, self.args.rnn_hidden_dim)

        if self.args.communicate:
            inputs = th.cat([self.hidden_states, self.comm_hidden_states], -1)
        else:
            inputs = self.hidden_states
        f = self.single_q(inputs.view(-1, self.input_shape))
        f = f.view(-1, self.n_agents, self.n_actions)

        input_i = inputs.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        input_j = inputs.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
        inputs = th.cat([input_i, input_j], dim=-1).view(-1, 2 * self.input_shape)
        g = self.pairwise_q(inputs)
        g = g.view(-1, self.n_agents, self.n_agents, self.n_actions, self.n_actions)
        g = (g + g.permute(0, 2, 1, 4, 3)) / 2.

        if select_graph == False:
            return f, g

        graphs = self.solver.solve(f, g, avail_actions, self.args.device)
        return f, g, graphs

    def communicate(self, graphs):
        # Exchange hidden state
        onehot_id = th.eye(self.n_agents).to(self.args.device)
        onehot_id = onehot_id.expand(graphs.shape[0], self.n_agents, self.n_agents)
        states = th.cat([self.comm_hidden_states, onehot_id], dim=2)

        index = (graphs * th.tensor([i for i in range(self.n_agents)]).to(self.args.device)) \
            .sum(-1, keepdim=True).long().repeat(1, 1, self.comm_input_shape)
        inputs = th.gather(states, dim=1, index=index)
        mask = graphs.sum(-1, keepdim=True)
        self.comm_hidden_states = (1 - mask) * self.comm_hidden_states + mask * \
            self.comm_unit.forward(inputs.view(-1, self.comm_input_shape), self.comm_hidden_states).view(graphs.shape[0], self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        if self.args.communicate:
            self.comm_hidden_states = self.comm_unit.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        if self.args.communicate:
            return itertools.chain(self.agent.parameters(), self.single_q.parameters(), self.pairwise_q.parameters(),
                                   self.comm_unit.parameters())
        return itertools.chain(self.agent.parameters(), self.single_q.parameters(), self.pairwise_q.parameters())

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.single_q.load_state_dict(other_mac.single_q.state_dict())
        self.pairwise_q.load_state_dict(other_mac.pairwise_q.state_dict())
        if self.args.communicate:
            self.comm_unit.load_state_dict(other_mac.comm_unit.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.single_q.cuda()
        self.pairwise_q.cuda()
        if self.args.communicate:
            self.comm_unit.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.single_q.state_dict(), "{}/single_q.th".format(path))
        th.save(self.pairwise_q.state_dict(), "{}/pairwise_q.th".format(path))
        if self.args.communicate:
            th.save(self.comm_unit.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.single_q.load_state_dict(th.load("{}/single_q.th".format(path), map_location=lambda storage, loc: storage))
        self.pairwise_q.load_state_dict(
            th.load("{}/pairwise_q.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.communicate:
            self.comm_unit.load_state_dict(
                th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    # ========================= Private methods =========================

    @staticmethod
    def _mlp(input, hidden_dims, output):
        dim = input
        layers = []
        for d in hidden_dims:
            layers.append(nn.Linear(dim, d))
            layers.append(nn.ReLU())
            dim = d
        layers.append(nn.Linear(dim, output))
        return nn.Sequential(*layers)