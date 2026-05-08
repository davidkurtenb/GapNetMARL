import math

#import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

import sys
sys.path.append(r'C:\Users\dk412\Desktop\David\Python Projects\C2MARA_HyperSwarm\src')

from models.hypernetwork import HyperNetwork, hyper_forward
from utils.capabilities_gapnet import CAP_DIM, NUM_TYPES, AGENT_TYPES

class HybridActorCritic(nn.Module):

    def __init__(
        self,
        type_obs_dims,
        type_action_specs,
        hidden_dim=64,
        hypernet_hidden_dim=64,
        hypernet_layers=2,
        gap_dim=8,
        activation="relu",
    ):
        super().__init__()
        self.type_obs_dims = type_obs_dims
        self.type_action_specs = type_action_specs
        self.hidden_dim = hidden_dim
        self.gap_dim = gap_dim
        self.act_fn = torch.tanh if activation == "tanh" else torch.relu

        cond_dim = CAP_DIM + NUM_TYPES + gap_dim

        self.encoder_first = nn.ModuleDict({
            atype: nn.Linear(obs_dim, hidden_dim)
            for atype, obs_dim in type_obs_dims.items()
        })
        self.encoder_second = nn.Linear(hidden_dim, hidden_dim)

        # Hnet Actors
        self._continuous_action_dim = None
        self._discrete_action_dim = None
        for atype, (space_type, act_dim) in type_action_specs.items():
            if space_type == "continuous":
                self._continuous_action_dim = act_dim
            else:
                self._discrete_action_dim = act_dim

        if self._continuous_action_dim is not None:
            self.continuous_actor_hypernet = HyperNetwork(
                cond_dim, hidden_dim, self._continuous_action_dim,
                hidden_dim=hypernet_hidden_dim, num_layers=hypernet_layers,
                use_layer_norm=True,
            )
            #Learnable log_std shared across all continuous agents)
            self.log_std = nn.Parameter(torch.zeros(self._continuous_action_dim))

        if self._discrete_action_dim is not None:
            self.discrete_actor_hypernet = HyperNetwork(
                cond_dim, hidden_dim, self._discrete_action_dim,
                hidden_dim=hypernet_hidden_dim, num_layers=hypernet_layers,
                use_layer_norm=True,
            )

        #Critic Hypernetwork
        #Shared across all agent types 
        self.critic_hypernet = HyperNetwork(
            cond_dim, hidden_dim, 1,
            hidden_dim=hypernet_hidden_dim, num_layers=hypernet_layers,
            use_layer_norm=True,
        )

        self._init_encoder_weights()

    def _init_encoder_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and "encoder" in name:
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

    def _encode(self, obs, agent_type):

        x = self.act_fn(self.encoder_first[agent_type](obs))
        x = self.act_fn(self.encoder_second(x))
        return x

    def forward(self, obs, agent_type, capability, gap_actor, gap_critic=None):
        # gap_actor conditions the actor hypernet; gap_critic conditions the critic.
        # Both receive gradients so the gap encoder learns from policy and value loss.
        # Defaults gap_critic to gap_actor when not provided separately.
        if gap_critic is None:
            gap_critic = gap_actor

        type_idx = AGENT_TYPES.index(agent_type)
        type_oh = torch.zeros(NUM_TYPES, device=obs.device)
        type_oh[type_idx] = 1.0
        type_oh = type_oh.unsqueeze(0).expand(obs.shape[0], -1)

        embedding = self._encode(obs, agent_type)
        cond_actor  = torch.cat([capability, type_oh, gap_actor],  dim=-1)
        cond_critic = torch.cat([capability, type_oh, gap_critic], dim=-1)
        space_type, _ = self.type_action_specs[agent_type]

        if space_type == "continuous":
            actor_w, actor_b = self.continuous_actor_hypernet(cond_actor)
            mean = hyper_forward(embedding, actor_w, actor_b)
            std = self.log_std.exp().expand_as(mean)
            action_params = (mean, std)
        else:
            actor_w, actor_b = self.discrete_actor_hypernet(cond_actor)
            logits = hyper_forward(embedding, actor_w, actor_b)
            action_params = logits

        # Critic value
        critic_w, critic_b = self.critic_hypernet(cond_critic)
        value = hyper_forward(embedding, critic_w, critic_b).squeeze(-1)

        return action_params, value

    def get_action_and_value(self, obs, capability=None, gap=None, agent_type=None):

        #if agent_type is None:
        #    agent_type = AGENT_TYPES[gap[0].argmax().item()]

        action_params, value = self.forward(obs, agent_type, capability, gap)

        space_type, _ = self.type_action_specs[agent_type]
        if space_type == "continuous":
            mean, std = action_params
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            return action.detach().cpu().squeeze(0).numpy(), log_prob.item(), value.item()
        else:
            logits = action_params
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs, actions, capability=None, gap=None, gap_critic=None, agent_type=None):

        action_params, values = self.forward(obs, agent_type, capability, gap, gap_critic)

        space_type, _ = self.type_action_specs[agent_type]
        if space_type == "continuous":
            mean, std = action_params
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            logits = action_params
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

        return log_probs, values, entropy
