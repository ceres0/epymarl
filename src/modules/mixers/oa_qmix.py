# Modified from EPyMARL for OA-QMIX and opponent-profile research.
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class OAQMixer(nn.Module):
    def __init__(self, args):
        super(OAQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.opp_profile_dim = int(getattr(args, "opp_profile_dim", 0) or 0)
        self.critic_input_dim = self.state_dim + self.opp_profile_dim
        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(
                self.critic_input_dim, self.embed_dim * self.n_agents
            )
            self.hyper_w_final = nn.Linear(self.critic_input_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.critic_input_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.critic_input_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        self.hyper_b_1 = nn.Linear(self.critic_input_dim, self.embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.critic_input_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states, opp_profile=None):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        if self.opp_profile_dim > 0:
            if opp_profile is None:
                opp_profile = states.new_zeros(states.size(0), self.opp_profile_dim)
            else:
                opp_profile = opp_profile.reshape(-1, self.opp_profile_dim)
            critic_input = th.cat([states, opp_profile], dim=-1)
        else:
            critic_input = states
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        w1 = th.abs(self.hyper_w_1(critic_input))
        b1 = self.hyper_b_1(critic_input)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        w_final = th.abs(self.hyper_w_final(critic_input))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(critic_input).view(-1, 1, 1)
        y = th.bmm(hidden, w_final) + v
        return y.view(bs, -1, 1)
