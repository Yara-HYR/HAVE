import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def reparameterize(self, mu, std):
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean) * self.factor
        sqrtvar_std = self.sqrtvar(std) * self.factor

        beta = self.reparameterize(mean, sqrtvar_mu)
        gamma = self.reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x

class fusion(nn.Module):
    def __init__(self,state_shape,device, n_agents,mixing_embed_dim,hypernet_embed,agent_own_state_size):
        super(fusion, self).__init__()

        # self.args = args
        self.n_agents = n_agents
        # tate_shape = obs_shape
        self.agent_own_state_size =agent_own_state_size
        self.state_dim = int(np.prod(state_shape))
        self.u_dim = int(np.prod(agent_own_state_size))

        self.n_query_embedding_layer1 = 64#args.n_query_embedding_layer1
        self.n_query_embedding_layer2 = 32#args.n_query_embedding_layer2
        self.n_key_embedding_layer1 = 32#args.n_key_embedding_layer1
        self.n_head_embedding_layer1 = 64#args.n_head_embedding_layer1
        self.n_head_embedding_layer2 = 4#args.n_head_embedding_layer2
        self.n_attention_head = 1#args.n_attention_head
        self.n_constrant_value = 32#args.n_constrant_value

        self.query_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.query_embedding_layers.append(nn.Sequential(nn.Linear(self.state_dim, self.n_query_embedding_layer1),
                                                           nn.ReLU(),
                                                           nn.Linear(self.n_query_embedding_layer1, self.n_query_embedding_layer2)).to(device))
        
        self.key_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.key_embedding_layers.append(nn.Linear(self.u_dim, self.n_key_embedding_layer1).to(device))


        self.scaled_product_value = np.sqrt(self.n_query_embedding_layer2)

        self.head_embedding_layer = nn.Sequential(nn.Linear(self.state_dim, self.n_head_embedding_layer1),
                                                  nn.ReLU(),
                                                  nn.Linear(self.n_head_embedding_layer1, self.n_head_embedding_layer2)).to(device)
        
        self.constrant_value_layer = nn.Sequential(nn.Linear(self.state_dim, self.n_constrant_value),
                                                  nn.ReLU(),
                                                  nn.Linear(self.n_constrant_value, 1)).to(device)




    def forward(self, agent_qs, states):
        # import pdb
        # pdb.set_trace()
        bs = agent_qs[0].size(0)
        states = states.reshape(-1, self.state_dim)
        us = self._get_us(states)
        agent_qs = th.stack(agent_qs,dim=2)#agent_qs.view(-1, 1, self.n_agents)

        q_lambda_list = []
        for i in range(self.n_attention_head):
            state_embedding = self.query_embedding_layers[i](states)
            u_embedding = self.key_embedding_layers[i](us)

            # shape: [-1, 1, state_dim]
            state_embedding = state_embedding.reshape(-1, 1, self.n_query_embedding_layer2)
            # shape: [-1, state_dim, n_agent]
            u_embedding = u_embedding.reshape(-1, self.n_agents, self.n_key_embedding_layer1)
            u_embedding = u_embedding.permute(0, 2, 1)

            # shape: [-1, 1, n_agent]
            raw_lambda = th.matmul(state_embedding, u_embedding) / self.scaled_product_value
            q_lambda = F.softmax(raw_lambda, dim=-1)

            q_lambda_list.append(q_lambda)

        # shape: [-1, n_attention_head, n_agent]
        q_lambda_list = th.stack(q_lambda_list, dim=1).squeeze(-2)

        # shape: [-1, n_agent, n_attention_head]
        q_lambda_list = q_lambda_list.permute(0, 2, 1)

        # shape: [-1, 1, n_attention_head]
        q_h = th.matmul(agent_qs, q_lambda_list)


        sum_q_h = q_h.sum(-1)
        sum_q_h = sum_q_h.reshape(-1, 1)

        c = self.constrant_value_layer(states)
        q_tot = sum_q_h + c
        q_tot = q_tot.view(bs, 1)
        return q_tot

    def _get_us(self, states):
        agent_own_state_size = self.agent_own_state_size
        with th.no_grad():
            us = states[:, :agent_own_state_size*self.n_agents].reshape(-1, agent_own_state_size)
        return us
