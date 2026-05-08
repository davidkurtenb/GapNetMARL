<p align="center">
  <img src="Gapnet_logo.png" alt="GapNet Logo" width="400"/>
</p>

# GapNet — Capability-Gap-Conditioned MARL for Heterogeneous Teams

GapNet is a multi-agent reinforcement learning framework for heterogeneous teams operating under attrition. It introduces a **Capability Gap Encoder** that continuously measures the gap between what a team can supply and what a mission requires, then conditions each agent's policy on that gap signal via a hypernetwork.

---

## Architecture

### Capability Gap Encoder
The encoder computes a team-level gap vector `g = R_eff - S` that summarizes how far the current active team is from fulfilling mission requirements.

**Supply aggregation** (attention-weighted DeepSets):
```
phi_j   = phi([c_j ; t_j])              per-agent embedding
alpha_j = softmax(phi_j · q / sqrt(d)) attention over active agents
S       = Σ alpha_j · phi_j            team supply estimate
```

**Requirement selection** (learned role bank):
```
dist_k  = ||R_k - S||         distance to each of K requirements
w_k     = softmax(-dist / tau) soft selection weights
R_eff   = Σ w_k * R_k         effective requirement
g       = R_eff - S            gap vector (GAP_DIM = 16)
```

Inactive agents (attrition) are masked to `-inf` before the softmax, so the supply estimate always reflects the *currently active* team.

### HybridActorCritic + Hypernetwork
A shared encoder backbone produces per-agent embeddings. A hypernetwork conditioned on `[capability ; type_onehot ; g]` generates agent-specific actor and critic weights, enabling zero-shot generalisation to unseen team compositions without retraining.

- Continuous action space: drones (Normal distribution, learned shared log-std)
- Discrete action space: observers, provisioners (Categorical)

---

## Training

**Algorithm:** PPO with GAE (`gamma=0.99`, `lambda=0.95`, `clip_eps=0.2`)

**Attrition curriculum:** Each episode, a fraction of agents per type are randomly dropped (always keeping at least one). The attrition fraction ramps linearly from 0 to `attrition_max` over `attrition_warmup` episodes, forcing the policy to learn robust coordination under degraded team conditions.

**Gap encoder update:** After each PPO epoch, the gap encoder is updated using a clipped importance-sampling policy gradient — the same objective as the PPO actor — plus a value function loss, giving the encoder direct signal about which team representations lead to better policies.

---

## Environments

Built on HeMAC (`HeMAC_v0`) via the PettingZoo API. Three fleet tiers of increasing complexity:

| Task | Area | Max Cycles | Agent Types | Obstacles | Notes |
|---|---|---|---|---|---|
| `simple_fleet` | 500×500 | 600 | Drones, Observers | None | Baseline |
| `mid_fleet` | 1000×1000 | 1200 | Drones, Observers | 1–2 | Larger teams |
| `complex_fleet` | 1500×1500 | 1200 | Drones, Observers, Provisioners | 4–8 | Rescue targets, full heterogeneity |

Each task includes multiple scenarios scaling from small fleets (e.g. `1q1o0p`) to large ones (e.g. `20q5o0p`).

---

## Repository Structure

```
src/
├── prod_code/
│   ├── train_gapnet_v3.py       # Timestep-budget training
│   └── train_gapnet_v3.2.py     # Episode-budget training (recommended)
├── models/
│   ├── hybrid_actor_critic.py   # HybridActorCritic + hypernetwork interface
│   └── hypernetwork.py          # HyperNetwork implementation
└── utils/
    ├── capabilities_gapnet_v3.py # CapabilityGapEncoder, capability extraction
    └── rollout_buffer.py         # RolloutBuffer with GAE
```

---

## Quickstart

```bash
pip install -r requirements.txt
python src/prod_code/train_gapnet_v3.2.py
```

Key parameters in `__main__`:

```python
task          = "simple_fleet"   # simple_fleet | mid_fleet | complex_fleet
num_episodes  = 3000             # episodes per scenario
attrition_max = 0.8              # maximum fraction of agents dropped per type
```

TensorBoard logs are written to `outputs/` per scenario and seed:

```bash
tensorboard --logdir outputs/
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `lr` / `lr_gap` | 3e-4 | Network / gap encoder learning rate |
| `hidden_dim` | 64 | Encoder hidden size |
| `GAP_DIM` | 16 | Gap vector dimension |
| `num_requirements` | 8 | Size of learned role bank |
| `temperature` | 2.0 | Softness of requirement selection |
| `attrition_warmup` | 500 | Episodes to ramp attrition to max |
| `entropy_coef` | 0.003 | PPO entropy bonus |
| `gap_pg_coef` | 0.1 | Weight of policy gradient term in gap update |
