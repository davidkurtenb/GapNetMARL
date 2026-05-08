from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################
# Raw capability extraction
################################################################

CAP_DIM = 6
AGENT_TYPES = ["drone", "observer", "provisioner"]
NUM_TYPES = len(AGENT_TYPES)

FUNC_DIM = 16
GAP_DIM = FUNC_DIM


def extract_capability_vector(agent_obj, agent_name):
    if "drone" in agent_name:
        cap = np.array([
            agent_obj.max_speed / 16.0,
            agent_obj.max_thrust / 4.0,
            agent_obj.max_charge / 10000.0,
            agent_obj.sensing_range / 100.0,
            float(agent_obj.carrying_capacity),
            agent_obj.time_factor,
        ], dtype=np.float32)
    elif "observer" in agent_name:
        cap = np.array([
            agent_obj.speed / 10.0,
            agent_obj.steering_angle / (np.pi / 4),
            agent_obj.sensor.hfov / np.pi if hasattr(agent_obj.sensor, 'hfov') else 0.0,
            agent_obj.sensor.sensing_range / 250.0,
            agent_obj.comm_range / 500.0,
            agent_obj.altitude / 100.0,
        ], dtype=np.float32)
    elif "provisioner" in agent_name:
        cap = np.array([
            agent_obj.max_speed / 10.0,
            agent_obj.max_thrust / 4.0,
            agent_obj.steering_angle / (np.pi / 4),
            agent_obj.sensor.hfov / np.pi if hasattr(agent_obj.sensor, 'hfov') else 0.0,
            agent_obj.sensor.sensing_range / 100.0,
            agent_obj.altitude / 100.0,
        ], dtype=np.float32)
    else:
        raise ValueError(f"Unknown agent type {agent_name}")

    assert cap.shape == (CAP_DIM,), f"Expected cap dim {CAP_DIM}, got {cap.shape}"
    return cap


def get_agent_type(agent_name: str) -> str:
    for t in AGENT_TYPES:
        if t in agent_name:
            return t
    raise ValueError(f"Unknown agent {agent_name}")


def get_type_onehot(agent_name: str) -> np.ndarray:
    oh = np.zeros(NUM_TYPES, dtype=np.float32)
    oh[AGENT_TYPES.index(get_agent_type(agent_name))] = 1.0
    return oh


def get_all_capabilities(env):
    raw_env = env
    while hasattr(raw_env, 'env'):
        raw_env = raw_env.env
    return {
        name: extract_capability_vector(raw_env.agents_list[i], name)
        for i, name in enumerate(env.possible_agents)
    }


#################################################################
# CapabilityGapEncoder v3 CHANGEFROM LAST version: attention-weighted team supply aggregation
#
# Supply aggregation:
#   phi_all = phi([c_j ; t_j])           for all j        (N, FUNC_DIM)
#   scores_j = (phi_j · q) / sqrt(d)     scaled dot-product  (N,)
#   alpha_j  = softmax(scores, inactive→-inf)               (N,)
#   S        = Σ alpha_j · phi_j                            (FUNC_DIM,)
#
# The learned query q asks "which agent's capability matters most for computing what this team can supply?" 
# helps when team composition changes via attrition, stops rare agent from getting higher attention than the  majority
#
# Requirement selection (inherited from v2):
#   dists_k  = ||R_k - S||
#   weights  = softmax(-dists / tau)
#   R_eff    = Σ weights_k * R_k
#   g        = R_eff - S
#################################################################

class CapabilityGapEncoder(nn.Module):

    def __init__(
        self,
        raw_cap_dim: int = CAP_DIM,
        num_types: int = NUM_TYPES,
        func_dim: int = FUNC_DIM,
        hidden: int = 64,
        learn_requirement: bool = True,
        num_requirements: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        in_dim = raw_cap_dim + num_types
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, func_dim),
        )
        self.func_dim = func_dim
        self.num_requirements = num_requirements
        self.temperature = temperature

        # Attention query; scaled init so dot-products start near unit variance
        self.query = nn.Parameter(torch.empty(func_dim))
        nn.init.normal_(self.query, mean=0.0, std=func_dim ** -0.5)

        # Role bank (K, FUNC_DIM)
        self.requirements = nn.Parameter(
            torch.empty(num_requirements, func_dim),
            requires_grad=learn_requirement,
        )
        nn.init.normal_(self.requirements, mean=0.0, std=0.1)

    ############################################
    # Internals
    #############################################

    def _phi(self, caps: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        return self.phi(torch.cat([caps, types], dim=-1))  # (N, FUNC_DIM)

    def _attn_weights(
        self,
        phi_all: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = (phi_all @ self.query) / (self.func_dim ** 0.5)  # (N,)
        scores = scores.masked_fill(active_mask == 0, float('-inf'))
        return torch.softmax(scores, dim=0)  # (N,)


    def encode_one(self, cap: torch.Tensor, type_onehot: torch.Tensor) -> torch.Tensor:
        return self.phi(torch.cat([cap, type_onehot], dim=-1))

    def team_supply(
        self,
        caps: torch.Tensor,
        types: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        #Attention-weighted supply S. Shape (FUNC_DIM,)
        phi_all = self._phi(caps, types)
        alpha = self._attn_weights(phi_all, active_mask)
        return (alpha.unsqueeze(-1) * phi_all).sum(dim=0)

    def select_requirement(self, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        dists = torch.norm(self.requirements - S.unsqueeze(0), dim=-1)  # (K,)
        weights = F.softmax(-dists / self.temperature, dim=0)
        R_eff = (weights.unsqueeze(-1) * self.requirements).sum(dim=0)
        return R_eff, weights

    def gap(
        self,
        caps: torch.Tensor,
        types: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        #g = R_eff - S. Shape(FUNC_DIM,)
        S = self.team_supply(caps, types, active_mask)
        R_eff, _ = self.select_requirement(S)
        return R_eff - S

    def gap_with_info(
        self,
        caps: torch.Tensor,
        types: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        phi_all = self._phi(caps, types)
        attn = self._attn_weights(phi_all, active_mask)
        S = (attn.unsqueeze(-1) * phi_all).sum(dim=0)
        R_eff, role_weights = self.select_requirement(S)
        return R_eff - S, S, role_weights, attn

    def forward(
        self,
        caps: torch.Tensor,
        types: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.gap(caps, types, active_mask)


################################################################
# Stack capabilitis 
################################################################

def stack_team_tensors(capabilities: dict, active: set, device) -> tuple:
    names = list(capabilities.keys())
    caps = np.stack([capabilities[n] for n in names], axis=0)
    types = np.stack([get_type_onehot(n) for n in names], axis=0)
    mask = np.array([1.0 if n in active else 0.0 for n in names], dtype=np.float32)
    return (
        torch.as_tensor(caps, dtype=torch.float32, device=device),
        torch.as_tensor(types, dtype=torch.float32, device=device),
        torch.as_tensor(mask, dtype=torch.float32, device=device),
        names,
    )
