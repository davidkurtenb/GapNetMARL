###### ADD TO jaxmarl/policies/gapnet_policies.py 

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal
from functools import partial

from jaxmarl.policies.policies import ScannedRNN, HyperNetwork

class GapEncoder(nn.Module):
    """
    DeepSets-style encoder: S = sum_j phi([c_j ; type_j]), gap = R - S.
    phi is a small MLP shared across agents. The type one-hot is concatenated
    to each capability vector before phi, matching CapabilityGapEncoder in HEMAC.
    Because S is a sum over a set, the output dim is fixed regardless of team size.
    """
    num_agents: int
    num_capabilities: int
    num_types: int = 3
    func_dim: int = 16
    phi_hidden: int = 64
    agent_type_indices: tuple = None 

    @nn.compact
    def __call__(self, all_caps, active_mask=None):

        ts, bs, _ = all_caps.shape

        caps_per_agent = all_caps.reshape(ts, bs, self.num_agents, self.num_capabilities)
        ego_cap = caps_per_agent[:, :, 0, :]  # (ts, bs, num_capabilities)

        flat_caps = caps_per_agent.reshape(ts * bs * self.num_agents, self.num_capabilities)

        # Build phi input: [cap ; type_onehot] if type indices are provided
        if self.agent_type_indices is not None:
            type_indices = jnp.array(self.agent_type_indices)  # (num_agents,)
            type_oh = jax.nn.one_hot(type_indices, self.num_types)  # (num_agents, num_types)
            type_oh_exp = jnp.broadcast_to(
                type_oh[None, None, :, :],
                (ts, bs, self.num_agents, self.num_types),
            )
            flat_types = type_oh_exp.reshape(ts * bs * self.num_agents, self.num_types)
            flat_input = jnp.concatenate([flat_caps, flat_types], axis=-1)
        else:
            flat_input = flat_caps
            type_oh = None

        # phi shared per-agent MLP into functional space
        phi_out = nn.Dense(self.phi_hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(flat_input)
        phi_out = nn.LayerNorm()(phi_out)
        phi_out = nn.relu(phi_out)
        phi_out = nn.Dense(self.func_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(phi_out)

        phi_out = phi_out.reshape(ts, bs, self.num_agents, self.func_dim)

        if active_mask is not None:
            phi_out = phi_out * active_mask[..., None]

        team_supply = phi_out.sum(axis=2)  # (ts, bs, func_dim)

        # Learnable requirement vector R —
        R = self.param('requirement', nn.initializers.normal(stddev=0.1), (self.func_dim,))
        gap = R[None, None, :] - team_supply  # (ts, bs, func_dim)

        if type_oh is not None:
            ego_type_oh = jnp.broadcast_to(
                type_oh[0][None, None, :], (ts, bs, self.num_types)
            )
        else:
            ego_type_oh = None

        return ego_cap, ego_type_oh, gap


class ActorGapHyperRNN(nn.Module):

    action_dim: int
    hidden_dim: int
    init_scale: float
    num_agents: int
    num_capabilities: int
    num_types: int = 3
    #GapNet-specific
    gap_func_dim: int = 16 # functional space dimension (matches HEMAC FUNC_DIM=16)
    gap_phi_hidden: int = 64 # phi MLP hidden width
    agent_type_indices: tuple = None
    #Hypernetwork config 
    hypernet_kwargs: dict = None

    def hyper_forward(self, in_dim, out_dim, target_in, hyper_in, time_steps, batch_size):
        hk = self.hypernet_kwargs

        num_weights = in_dim * out_dim
        weight_hypernet = HyperNetwork(
            hidden_dim=hk["HIDDEN_DIM"],
            output_dim=num_weights,
            init_scale=hk["INIT_SCALE"],
            num_layers=hk["NUM_LAYERS"],
            use_layer_norm=hk["USE_LAYER_NORM"],
        )
        weights = weight_hypernet(hyper_in).reshape(time_steps, batch_size, in_dim, out_dim)

        bias_hypernet = HyperNetwork(
            hidden_dim=hk["HIDDEN_DIM"],
            output_dim=out_dim,
            init_scale=0,
            num_layers=hk["NUM_LAYERS"],
            use_layer_norm=hk["USE_LAYER_NORM"],
        )
        biases = bias_hypernet(hyper_in).reshape(time_steps, batch_size, 1, out_dim)

        target_out = jnp.matmul(target_in[:, :, None, :], weights) + biases
        target_out = target_out.squeeze(axis=2)
        return target_out

    @nn.compact
    def __call__(self, hidden, x, active_mask=None):

        orig_obs, dones = x
        time_steps, batch_size, _ = orig_obs.shape

        dim_capabilities = self.num_agents * self.num_capabilities

        obs = orig_obs[:, :, :-dim_capabilities]
        all_caps = orig_obs[:, :, -dim_capabilities:]

        gap_encoder = GapEncoder(
            num_agents=self.num_agents,
            num_capabilities=self.num_capabilities,
            num_types=self.num_types,
            func_dim=self.gap_func_dim,
            phi_hidden=self.gap_phi_hidden,
            agent_type_indices=self.agent_type_indices,
        )
        ego_cap, ego_type_oh, gap = gap_encoder(all_caps, active_mask)

        #Hypernetwork 
        if ego_type_oh is not None:
            hyper_cond = jnp.concatenate([ego_cap, ego_type_oh, gap], axis=-1)
        else:
            hyper_cond = jnp.concatenate([ego_cap, gap], axis=-1)

        #Shared encoder 
        embedding = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        #RNN 
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        action_logits = self.hyper_forward(
            self.hidden_dim, self.action_dim,
            embedding, hyper_cond,
            time_steps, batch_size,
        )

        pi = distrax.Categorical(logits=action_logits)
        return hidden, pi
