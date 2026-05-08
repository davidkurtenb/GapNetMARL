#####################################################################
# GapNet v3.2
# Same as v3 with one change: episode-based termination only.
# Removed global_step / total_timesteps — training runs for exactly
# num_episodes per scenario, giving equal gradient updates across
# all fleet sizes regardless of team size or episode length.
#####################################################################

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import imageio
from hemac import HeMAC_v0

sys.path.append(r'C:\Users\dk412\Desktop\David\Python Projects\C2MARA_HyperSwarm\src')

from models.hybrid_actor_critic import HybridActorCritic
from utils.rollout_buffer import RolloutBuffer
from utils.capabilities_gapnet_v3 import (
    CapabilityGapEncoder,
    get_agent_type,
    get_all_capabilities,
    get_type_onehot,
    stack_team_tensors,
    CAP_DIM,
    NUM_TYPES,
    FUNC_DIM,
    GAP_DIM,
)

#############################################################
# Environment Configs
#############################################################

SIMPLE_FLEET_SCENARIOS = {
    "1q1o0p":  dict(n_drones=1,  n_observers=1, n_provisioners=0),
    "3q1o0p":  dict(n_drones=3,  n_observers=1, n_provisioners=0),
    "5q2o0p":  dict(n_drones=5,  n_observers=2, n_provisioners=0),
}

MID_FLEET_SCENARIOS = {
    "3q1o0p":  dict(n_drones=3,  n_observers=1, n_provisioners=0),
    "10q3o0p":  dict(n_drones=10,  n_observers=3, n_provisioners=0),
    "20q5o0p":  dict(n_drones=20,  n_observers=5, n_provisioners=0),
}

COMPLEX_FLEET_SCENARIOS = {
    "3q1o2p":  dict(n_drones=3,  n_observers=1, n_provisioners=1),
    "5q2o1p":  dict(n_drones=5,  n_observers=2, n_provisioners=1),
    "5q1o2p":  dict(n_drones=5,  n_observers=1, n_provisioners=2),
    "10q2o2p": dict(n_drones=10, n_observers=2, n_provisioners=2),
    "20q3o5p": dict(n_drones=20, n_observers=3, n_provisioners=5),
}

TASK_SCENARIOS = {
    "simple_fleet":  SIMPLE_FLEET_SCENARIOS,
    "mid_fleet":     MID_FLEET_SCENARIOS,
    "complex_fleet": COMPLEX_FLEET_SCENARIOS,
}

#############################################################
# Environment Config Task KWARGS simple,mid,complex Fleet
#############################################################

def get_task_kwargs(scenario, task):
    if task not in TASK_SCENARIOS:
        raise ValueError(f"Unknown task: {task}")
    counts = TASK_SCENARIOS[task][scenario]

    if task == 'complex_fleet':
        return dict(
            time_factor=1,
            area_size=(1500, 1500),
            max_cycles=1200,
            render_ratio=0.6,
            n_observers=counts["n_observers"],
            n_drones=counts["n_drones"],
            n_provisioners=counts["n_provisioners"],
            min_obstacles=4,
            max_obstacles=8,
            rescuing_targets=True,
            observer_comm_range=200,
            patrol_config={
                "benchmark": True,
                "area": [
                    (100, 100), (500, 50), (900, 200),
                    (950, 600), (700, 900), (300, 950), (50, 600),
                ],
            },
            poi_config=[
                {"speed": 1.5, "dimension": [8, 8],  "spawn_mode": "random"},
                {"speed": 2.0, "dimension": [10, 10], "spawn_mode": "random"},
                {"speed": 0.8, "dimension": [6, 6],  "spawn_mode": "random"},
            ],
            drone_config={
                "drones_starting_pos": [],
                "drone_ui_dimension": 16,
                "drone_max_speed": 10,
                "drone_max_charge": 400,
                "discrete_action_space": False,
            },
            drone_sensor={"model": "RoundCamera", "params": {"sensing_range": 30}},
            observer_sensor={"model": "ForwardFacingCamera",
                            "params": {"hfov": np.pi / 6, "sensing_range": 100}},
            provisioner_sensor={"model": "ForwardFacingCamera",
                                "params": {"hfov": np.pi / 2, "sensing_range": 50}},
        )
    elif task == 'mid_fleet':
        return dict(
            time_factor=1,
            area_size=(1000, 1000),
            max_cycles=1200,
            render_ratio=0.6,
            n_observers=counts["n_observers"],
            n_drones=counts["n_drones"],
            n_provisioners=counts["n_provisioners"],
            min_obstacles=1,
            max_obstacles=2,
            rescuing_targets=False,
            observer_comm_range=250,
            patrol_config={
                "benchmark": True,
                "area": [
                    (100, 100), (500, 50), (900, 200),
                    (950, 600), (700, 900), (300, 950), (50, 600),
                ],
            },
            poi_config=[
                {"speed": 1.5, "dimension": [8, 8],  "spawn_mode": "random"},
                {"speed": 2.0, "dimension": [10, 10], "spawn_mode": "random"},
                {"speed": 0.8, "dimension": [6, 6],  "spawn_mode": "random"},
            ],
            drone_config={
                "drones_starting_pos": [],
                "drone_ui_dimension": 16,
                "drone_max_speed": 10,
                "drone_max_charge": 500,
                "discrete_action_space": False,
            },
            drone_sensor={"model": "RoundCamera", "params": {"sensing_range": 40}},
            observer_sensor={"model": "ForwardFacingCamera",
                            "params": {"hfov": np.pi / 6, "sensing_range": 300}},
        )
    elif task == 'simple_fleet':
        return dict(
            time_factor=1,
            area_size=(500, 500),
            max_cycles=600,
            render_ratio=0.6,
            n_observers=counts["n_observers"],
            n_drones=counts["n_drones"],
            n_provisioners=counts["n_provisioners"],
            min_obstacles=0,
            max_obstacles=0,
            rescuing_targets=False,
            observer_comm_range=300,
            patrol_config={
                'benchmark': True,
                'area': [(100, 100), (400, 100), (480, 250), (400, 480), (100, 480)],
            },
            poi_config=[
                {"speed": 1.5, "dimension": [8, 8],  "spawn_mode": "random"},
                {"speed": 2.0, "dimension": [10, 10], "spawn_mode": "random"},
                {"speed": 0.8, "dimension": [6, 6],  "spawn_mode": "random"},
            ],
            drone_config={
                "drones_starting_pos": [],
                "drone_ui_dimension": 16,
                "drone_max_speed": 10,
                "drone_max_charge": 9999,
                "discrete_action_space": False,
            },
            drone_sensor={"model": "RoundCamera", "params": {"sensing_range": 50}},
            observer_sensor={"model": "ForwardFacingCamera",
                            "params": {"hfov": np.pi / 6, "sensing_range": 200}},
        )


#############################################################
# Reward normalization — Welford online algorithm
# Divides stored rewards by running std so value function never
# sees raw returns of ±3000, which cause catastrophic TD errors.
#############################################################

class RunningMeanStd:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return max((self.M2 / self.n) ** 0.5, 1e-4)

    def normalize(self, x: float) -> float:
        return float(np.clip(x / self.std, -10.0, 10.0))


#############################################################
# Attrition curriculum
#############################################################

def sample_active_set(possible_agents, attrition_frac, rng):
    if attrition_frac <= 0.0:
        return set(possible_agents)

    by_type = {}
    for a in possible_agents:
        by_type.setdefault(get_agent_type(a), []).append(a)

    active = set()
    for atype, members in by_type.items():
        n = len(members)
        n_drop = int(round(attrition_frac * n))
        n_drop = min(n_drop, n - 1)  # always keep at least one
        keep_idx = rng.choice(n, size=n - n_drop, replace=False)
        for k in keep_idx:
            active.add(members[k])
    return active


def attrition_schedule(episode, warmup, max_frac):
    if warmup <= 0:
        return max_frac
    return float(min(max_frac, max_frac * episode / warmup))

#############################################################
# Training
############################################################
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    log_dir = os.path.normpath(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"Logging to {log_dir}")

    env_kwargs = get_task_kwargs(args.scenario, args.task)
    env = HeMAC_v0.env(render_mode=args.render_mode, **env_kwargs)

    env.reset(seed=0)
    type_obs_dims, type_action_specs = {}, {}
    for agent in env.possible_agents:
        atype = get_agent_type(agent)
        if atype not in type_obs_dims:
            obs_space = env.observation_space(agent)
            act_space = env.action_space(agent)
            type_obs_dims[atype] = obs_space.shape[0]
            continuous = hasattr(act_space, "shape") and len(act_space.shape) > 0
            type_action_specs[atype] = (
                ("continuous", act_space.shape[0]) if continuous
                else ("discrete", act_space.n)
            )
    print("Agent type spaces:")
    for atype in type_obs_dims:
        print(f"  {atype}: obs_dim={type_obs_dims[atype]}, action={type_action_specs[atype]}")

    network = HybridActorCritic(
        type_obs_dims=type_obs_dims,
        type_action_specs=type_action_specs,
        hidden_dim=args.hidden_dim,
        hypernet_hidden_dim=args.hypernet_hidden_dim,
        hypernet_layers=args.hypernet_layers,
        gap_dim=GAP_DIM,
    ).to(device)

    gap_encoder = CapabilityGapEncoder(
        num_requirements=8,
        temperature=2.0,
    ).to(device)

    network_optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    gap_optimizer = torch.optim.Adam(gap_encoder.parameters(), lr=args.lr_gap)

    buffers = {agent_name: RolloutBuffer() for agent_name in env.possible_agents}

    type_to_agents = {}
    for agent_name in env.possible_agents:
        atype = get_agent_type(agent_name)
        type_to_agents.setdefault(atype, []).append(agent_name)

    act_highs = {}
    for agent in env.possible_agents:
        atype = get_agent_type(agent)
        if type_action_specs[atype][0] == "continuous":
            act_highs[atype] = float(env.action_space(agent).high[0])

    param_count = sum(p.numel() for p in network.parameters()) + \
                  sum(p.numel() for p in gap_encoder.parameters())
    print(f"Total parameters (hypernet and gap encoder): {param_count:,}")

    rng = np.random.default_rng(args.seed)
    reward_history = {atype: [] for atype in type_obs_dims}
    reward_rms = RunningMeanStd()  # normalizes stored rewards to prevent ±3000-scale TD errors

    def save_checkpoint(episode):
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"{args.task}_GapNET_PPO.pt")
        torch.save({
            "network_state_dict": network.state_dict(),
            "gap_encoder_state_dict": gap_encoder.state_dict(),
            "network_optimizer_state_dict": network_optimizer.state_dict(),
            "gap_optimizer_state_dict": gap_optimizer.state_dict(),
            "type_obs_dims": type_obs_dims,
            "type_action_specs": type_action_specs,
            "hidden_dim": args.hidden_dim,
            "hypernet_hidden_dim": args.hypernet_hidden_dim,
            "hypernet_layers": args.hypernet_layers,
            "gap_dim": GAP_DIM,
            "func_dim": FUNC_DIM,
            "num_requirements": gap_encoder.requirements.shape[0],
            "episode": episode,
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

    for episode in range(1, args.num_episodes + 1):
        env.reset(seed=episode)
        capabilities = get_all_capabilities(env)

        frac = attrition_schedule(episode, args.attrition_warmup, args.attrition_max)
        active = sample_active_set(env.possible_agents, frac, rng)
        writer.add_scalar("attrition/fraction", frac, episode)
        writer.add_scalar("attrition/n_active", len(active), episode)

        caps_t, types_t, active_mask, names = stack_team_tensors(
            capabilities, active, device
        )

        episode_rewards = {atype: 0.0 for atype in type_obs_dims}
        episode_steps = 0
        episode_pois_found = 0
        pending = {}
        last_values = {a: 0.0 for a in env.possible_agents}

        with torch.no_grad():
            g_episode, _, weights, attn_w = gap_encoder.gap_with_info(caps_t, types_t, active_mask)
            for k, w in enumerate(weights):
                writer.add_scalar(f"gap/role_weight_{k}", w.item(), episode)
            writer.add_scalar("gap/role_entropy", -(weights * weights.log().clamp(min=-10)).sum().item(), episode)
            writer.add_scalar("gap/attn_entropy", -(attn_w * attn_w.clamp(min=1e-8).log()).sum().item(), episode)
            writer.add_scalar("gap/top_attn_agent", attn_w.argmax().item(), episode)
            g_batch = g_episode.unsqueeze(0)  # (1, GAP_DIM)
            g_episode_cached = g_episode.clone()

        for agent_name in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            atype = get_agent_type(agent_name)
            episode_steps += 1

            if agent_name == env.possible_agents[0] and info.get("success", False):
                episode_pois_found += 1

            if agent_name in pending:
                prev = pending.pop(agent_name)
                prev_obs, prev_act, prev_lp, prev_val, prev_cap = prev
                done = termination

                if agent_name in active:
                    reward_rms.update(reward)
                    buffers[agent_name].store(prev_obs, prev_act, prev_lp,
                                              reward_rms.normalize(reward), prev_val, done, prev_cap)
                    episode_rewards[atype] += reward  # raw reward for logging

            if termination or truncation:
                if episode <= 5:
                    reason = "TERM" if termination else "TRUNC"
                    print(f"  [diag ep{episode}] {agent_name} {reason} step={episode_steps} r={reward:.2f} info={info}")
                if truncation and agent_name in active:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    cap_b = torch.as_tensor(capabilities[agent_name], dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        _, _, last_v = network.get_action_and_value(
                            obs_t, capability=cap_b, gap=g_batch, agent_type=atype
                        )
                    last_values[agent_name] = last_v
                env.step(None)
                continue

            if agent_name not in active:
                if type_action_specs[atype][0] == "continuous":
                    env.step(np.zeros(type_action_specs[atype][1], dtype=np.float32))
                else:
                    env.step(0)
                continue

            cap = capabilities[agent_name]
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            cap_b = torch.as_tensor(cap, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, value = network.get_action_and_value(
                    obs_t,
                    capability=cap_b,
                    gap=g_batch, #replaces type_onehot
                    agent_type=atype,
                )

            if type_action_specs[atype][0] == "continuous":
                env_action = np.clip(action, -act_highs[atype], act_highs[atype])
            else:
                env_action = action

            pending[agent_name] = (obs.copy(), action, log_prob, value, cap)  # raw action for PPO
            env.step(env_action)

        #PPO update
        total_loss_info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        total_updates = 0
        per_type_data = {}

        for atype, agent_names in type_to_agents.items():
            #Compute GAE per-agent
            all_obs, all_caps, all_acts, all_old_lp = [], [], [], []
            all_adv, all_ret = [], []

            for agent_name in agent_names:
                buf = buffers[agent_name]
                if len(buf) == 0:
                    continue
                adv, ret = buf.compute_gae(args.gamma, args.gae_lambda, last_value=last_values[agent_name])
                all_obs.extend(buf.obs)
                all_caps.extend(buf.capabilities)
                all_acts.extend(buf.actions)
                all_old_lp.extend(buf.log_probs)
                all_adv.append(adv)
                all_ret.append(ret)
                buf.clear()

            if not all_obs:
                continue

            obs_b  = torch.as_tensor(np.array(all_obs), dtype=torch.float32, device=device)
            caps_b = torch.as_tensor(np.array(all_caps), dtype=torch.float32, device=device)
            space_type = type_action_specs[atype][0]
            if space_type == "continuous":
                act_b = torch.as_tensor(np.array(all_acts), dtype=torch.float32, device=device)
            else:
                act_b = torch.as_tensor(np.array(all_acts), dtype=torch.long, device=device)
            old_lp_b = torch.as_tensor(np.array(all_old_lp), dtype=torch.float32, device=device)
            adv_b = torch.as_tensor(np.concatenate(all_adv), dtype=torch.float32, device=device)
            ret_b = torch.as_tensor(np.concatenate(all_ret), dtype=torch.float32, device=device)
            if len(adv_b) < 2:

                continue
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
            adv_b = adv_b.clamp(-10.0, 10.0)

            per_type_data[atype] = (obs_b, caps_b, act_b, ret_b, adv_b)
            n = len(obs_b)
            for _ in range(args.update_epochs):
                indices = np.random.permutation(n)
                for start in range(0, n, args.minibatch_size):
                    end = min(start + args.minibatch_size, n)
                    mb = indices[start:end]

                    g_mb = g_episode_cached.unsqueeze(0).expand(len(mb), -1)

                    new_lp, values, entropy = network.evaluate(
                        obs_b[mb], act_b[mb],
                        capability=caps_b[mb],
                        gap=g_mb,
                        gap_critic=g_mb,
                        agent_type=atype,
                    )

                    ratio = (new_lp - old_lp_b[mb]).exp()
                    pg1 = -adv_b[mb] * ratio
                    pg2 = -adv_b[mb] * torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                    pg_loss = torch.max(pg1, pg2).mean()
                    vf_loss = ((values - ret_b[mb]) ** 2).mean()
                    loss = pg_loss + args.value_coef * vf_loss - args.entropy_coef * entropy.mean()

                    network_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm)
                    network_optimizer.step()

                    total_loss_info["policy_loss"] += pg_loss.item()
                    total_loss_info["value_loss"] += vf_loss.item()
                    total_loss_info["entropy"] += entropy.mean().item()
                    total_updates += 1

        # Gap encoder update: multiple steps per episode using value + policy gradient,
        # after PPO epochs. The policy gradient gives the gap encoder direct signal about which team representations lead tobetter actions
        if per_type_data and episode > args.gap_warmup_episodes:
            n_types = len(per_type_data)
            for _ in range(args.gap_update_epochs):
                gap_optimizer.zero_grad()
                for atype, (obs_b, caps_b, act_b, ret_b, adv_b) in per_type_data.items():
                    n = len(obs_b)
                    mb_idx = np.random.choice(n, min(n, args.minibatch_size * 2), replace=False)
                    g = gap_encoder.gap(caps_t, types_t, active_mask)
                    g_mb = g.unsqueeze(0).expand(len(mb_idx), -1)
                    new_lp, values, _ = network.evaluate(
                        obs_b[mb_idx], act_b[mb_idx],
                        capability=caps_b[mb_idx],
                        gap=g_mb,
                        gap_critic=g_mb,
                        agent_type=atype,
                    )
                    gap_vf_loss = ((values - ret_b[mb_idx]) ** 2).mean() / n_types
                    gap_pg_loss = -(new_lp * adv_b[mb_idx]).mean() / n_types
                    (gap_vf_loss + args.gap_pg_coef * gap_pg_loss).backward()
                nn.utils.clip_grad_norm_(gap_encoder.parameters(), args.max_grad_norm)
                gap_optimizer.step()
            network_optimizer.zero_grad()

        with torch.no_grad():
            g_log = gap_encoder.gap(caps_t, types_t, active_mask)
            writer.add_scalar("gap/norm",  g_log.norm().item(), episode)
            writer.add_scalar("gap/mean",  g_log.mean().item(), episode)
            writer.add_scalar("gap/R_norm", gap_encoder.requirements.norm().item(), episode)

        for atype in type_obs_dims:
            writer.add_scalar(f"{atype}/episode_reward", episode_rewards[atype], episode)
            reward_history[atype].append(episode_rewards[atype])

        if total_updates > 0:
            writer.add_scalar("train/policy_loss", total_loss_info["policy_loss"] / total_updates, episode)
            writer.add_scalar("train/value_loss", total_loss_info["value_loss"] / total_updates, episode)
            writer.add_scalar("train/entropy", total_loss_info["entropy"] / total_updates, episode)

        total_reward = sum(episode_rewards.values())
        writer.add_scalar("total/episode_steps", episode_steps, episode)
        writer.add_scalar("total/episode_reward", total_reward, episode)
        writer.add_scalar("task/pois_found", episode_pois_found, episode)
        buf_transitions = sum(len(buffers[a]) for a in buffers)
        writer.add_scalar("debug/buffer_transitions", buf_transitions, episode)
        if episode_steps < 10:
            print(f"[WARN] Episode {episode}: suspiciously short ({episode_steps} steps).")

        if episode % args.log_interval == 0:
            rw_str = ", ".join(f"{t}: {r:.2f}" for t, r in episode_rewards.items())
            print(f"Ep {episode}/{args.num_episodes} | steps={episode_steps} | pois={episode_pois_found} | "
                  f"attrit={frac:.2f} | {rw_str}")

        if episode % args.save_interval == 0 or episode == args.num_episodes:
            save_checkpoint(episode)

    #Final score mean total reward over last 100 episodes
    n_hist = len(next(iter(reward_history.values())))
    total_reward_history = [
        sum(reward_history[atype][ep] for atype in reward_history)
        for ep in range(n_hist)
    ]
    window_final = min(100, n_hist)
    final_score = float(np.mean(total_reward_history[-window_final:]))
    writer.add_scalar("total/final_score", final_score, episode)
    print(f"Final score (mean total reward, last {window_final} eps): {final_score:.2f}")

    #plots
    plots_dir = os.path.join(os.path.dirname(args.log_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    n_eps = max(len(v) for v in reward_history.values()) if reward_history else 0
    episodes = range(1, n_eps + 1)

    for atype, rewards in reward_history.items():
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(list(episodes)[:len(rewards)], rewards, alpha=0.4, label="raw")
        window = min(20, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax.plot(range(window, len(rewards)+1), smoothed, label=f"{window}-ep avg")
        ax.set_title(f"{atype} Episode Reward (GapNet)")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.legend()
        fig.savefig(os.path.join(plots_dir, f"reward_{atype}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    total_rewards = [sum(reward_history[t][i] for t in reward_history if i < len(reward_history[t]))
                     for i in range(n_eps)]
    fig, ax = plt.subplots(figsize=(10, 4))
    window = min(20, n_eps)
    for atype, rewards in reward_history.items():
        eps = list(range(1, len(rewards) + 1))
        ax.plot(eps, rewards, alpha=0.25)
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
            ax.plot(range(window, len(rewards)+1), smoothed, label=atype)
    if total_rewards:
        ax.plot(list(episodes), total_rewards, alpha=0.25, color="black")
        if n_eps >= window:
            smoothed_total = np.convolve(total_rewards, np.ones(window)/window, mode="valid")
            ax.plot(range(window, n_eps+1), smoothed_total, color="black", linewidth=2, label="total")
    ax.set_title("Episode Reward over Training (GapNet)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.legend()
    fig.savefig(os.path.join(plots_dir, "reward_all.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    env.close()
    writer.close()
    print("Training complete")

def record_video(args, checkpoint_path, output_path, scenario, seed=0, max_steps=1200):
    import copy
    eval_args = copy.copy(args)
    eval_args.scenario = scenario
    eval_args.seed = seed
    eval_args.render_mode = None # no display during eval

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_kwargs = get_task_kwargs(scenario, args.task)
    render_env = HeMAC_v0.env(render_mode="rgb_array", **env_kwargs)
    render_env.reset(seed=seed)

    type_obs_dims, type_action_specs = {}, {}
    for agent in render_env.possible_agents:
        atype = get_agent_type(agent)
        if atype not in type_obs_dims:
            obs_space = render_env.observation_space(agent)
            act_space = render_env.action_space(agent)
            type_obs_dims[atype] = obs_space.shape[0]
            continuous = hasattr(act_space, "shape") and len(act_space.shape) > 0
            type_action_specs[atype] = (
                ("continuous", act_space.shape[0]) if continuous
                else ("discrete", act_space.n)
            )

    #Load
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network = HybridActorCritic(
        type_obs_dims=ckpt["type_obs_dims"],
        type_action_specs=ckpt["type_action_specs"],
        hidden_dim=ckpt["hidden_dim"],
        hypernet_hidden_dim=ckpt["hypernet_hidden_dim"],
        hypernet_layers=ckpt["hypernet_layers"],
        gap_dim=ckpt["gap_dim"],
    ).to(device)
    network.load_state_dict(ckpt["network_state_dict"])
    network.eval()

    gap_encoder = CapabilityGapEncoder(
        num_requirements=ckpt.get("num_requirements", 8),
        temperature=2.0,
    ).to(device)
    gap_encoder.load_state_dict(ckpt["gap_encoder_state_dict"])
    gap_encoder.eval()

    act_highs = {}
    for agent in render_env.possible_agents:
        atype = get_agent_type(agent)
        if type_action_specs[atype][0] == "continuous":
            act_highs[atype] = float(render_env.action_space(agent).high[0])

    capabilities = get_all_capabilities(render_env)
    caps_t, types_t, active_mask, _ = stack_team_tensors(
        capabilities, set(render_env.possible_agents), device
    )

    with torch.no_grad():
        g_video = gap_encoder.gap(caps_t, types_t, active_mask)
    g_batch = g_video.unsqueeze(0)

    frames = []
    step = 0

    frame = render_env.render()
    if frame is not None:
        frames.append(frame)

    for agent_name in render_env.agent_iter():
        obs, _, termination, truncation, _ = render_env.last()
        atype = get_agent_type(agent_name)

        if termination or truncation:
            render_env.step(None)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            cap = capabilities[agent_name]
            cap_b = torch.as_tensor(cap, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, _, _ = network.get_action_and_value(
                    obs_t, capability=cap_b, gap=g_batch, agent_type=atype
                )

            if type_action_specs[atype][0] == "continuous":
                action = np.clip(action, -act_highs[atype], act_highs[atype])

            render_env.step(action)

        frame = render_env.render()
        if frame is not None:
            frames.append(frame)

        step += 1
        if step >= max_steps * len(render_env.possible_agents):
            break

    render_env.close()

    if frames:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        imageio.mimsave(output_path, frames, fps=15)
        print(f"Saved {len(frames)}-frame video to {output_path}")
    else:
        print("garbage")

###################################################
# MAIN
###################################################
if __name__ == "__main__":
    task = "simple_fleet"
    num_episodes = 10000
    attrition_max = 0.4
    version = "3.2"

    run_time = datetime.now().strftime("%Y%m%d%H%M%S")
    base_dir = rf"C:\Users\dk412\Desktop\David\Python Projects\C2MARA_HyperSwarm\outputs\gapnet_v{version}_{task}_attrition{attrition_max}_episodes{num_episodes}_{run_time}"

    args = argparse.Namespace(
        num_episodes=num_episodes,
        lr=3e-4,
        lr_gap=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.003,
        value_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=4,
        gap_update_epochs=4,
        gap_pg_coef=0.1,
        minibatch_size=64,
        hidden_dim=64,
        hypernet_hidden_dim=64,
        hypernet_layers=2,
        phi_hidden=64,
        attrition_max=attrition_max,
        attrition_warmup=500,
        log_dir="",
        save_dir="",
        render_dir="",
        save_interval=100,
        log_interval=10,
        task=task,
        scenario="",
        render_mode=True,
        seed=42,
        gap_warmup_episodes=0,
    )

    scenarios_to_run = list(TASK_SCENARIOS[task].keys())

    for scenario in scenarios_to_run:
        print(f"=" * 80)
        print(f"Running task: {task}")
        print(f"=" * 80)
        for seed in range(3):
            print(f"\n{'='*60}")
            print(f"  Scenario: {scenario}  |  Seed: {seed}")
            print(f"{'='*60}\n")

            args.scenario = scenario
            args.seed = seed
            args.log_dir = os.path.join(base_dir, "tensorboard_logs", scenario, f"seed_{seed}")
            args.save_dir = os.path.join(base_dir, "training_checkpoints", scenario, f"seed_{seed}")
            args.render_dir = os.path.join(base_dir, "renders", scenario, f"seed_{seed}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            train(args)

            os.makedirs(args.render_dir, exist_ok=True)

            if args.render_mode:
                record_video(
                    args,
                    checkpoint_path=os.path.join(args.save_dir, f"{args.task}_GapNET_PPO.pt"),
                    output_path=os.path.join(args.render_dir, f"{scenario}_seed_{seed}.mp4"),
                    scenario=scenario,
                    seed=seed,
                )

    print("FINISHED TRAINING")
