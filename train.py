"""Train the shared DQN, then compare it with a pure A* baseline.

    python train.py --episodes 600 --eval-episodes 100
"""
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.env import MultiRobotEnv, STAY


def run_episode(env, policy, agent=None, seed=None):
    """policy(env, i, obs) -> action. If `agent` is given, learn online."""
    obs = env.reset(seed=seed)
    total = 0.0
    while True:
        actions = [policy(env, i, o) for i, o in enumerate(obs)]
        next_obs, rewards, term, trunc, info = env.step(actions)
        if agent is not None:
            for i, was_active in enumerate(info["active"]):
                if was_active:
                    agent.remember(obs[i], actions[i], rewards[i], next_obs[i], term)
            agent.train_step()
        total += sum(r for r, a in zip(rewards, info["active"]) if a)
        obs = next_obs
        if term or trunc:
            return {"return": total, "steps": info["steps"], "completed": info["completed"],
                    "collisions": info["collisions"], "success": float(term)}


def astar_policy(env, i, obs):
    r = env.robots[i]
    return env.hint_action(r) if r.task is not None else STAY


def make_dqn_policy(agent, greedy):
    def policy(env, i, obs):
        if env.robots[i].task is None:
            return STAY
        return agent.select_action(obs, greedy=greedy)
    return policy


def evaluate(env, policy, n, seed0=10_000):
    rows = [run_episode(env, policy, seed=seed0 + k) for k in range(n)]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def moving_avg(x, w=20):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--robots", type=int, default=2)
    args = p.parse_args()

    import torch
    from agents.agents import DQNAgent
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    env = MultiRobotEnv(n_robots=args.robots, seed=args.seed)
    agent = DQNAgent(env.obs_dim, env.n_actions)
    train_policy = make_dqn_policy(agent, greedy=False)

    history = []
    for ep in range(1, args.episodes + 1):
        res = run_episode(env, train_policy, agent=agent)
        history.append(res)
        if ep % 25 == 0:
            last = history[-25:]
            print(f"ep {ep:4d} | eps {agent.epsilon():.2f} | "
                  f"return {np.mean([h['return'] for h in last]):7.1f} | "
                  f"done {np.mean([h['success'] for h in last]):.2f} | "
                  f"coll {np.mean([h['collisions'] for h in last]):.1f}")
    agent.save("models/dqn.pt")

    print("\nEvaluating on unseen seeds (greedy DQN vs A*-only)...")
    results = {
        "A* only": evaluate(env, astar_policy, args.eval_episodes),
        "A* hint + DQN": evaluate(env, make_dqn_policy(agent, greedy=True), args.eval_episodes),
    }
    print(f"\n{'policy':16s} {'steps':>8s} {'tasks':>8s} {'collisions':>11s} {'all done':>9s}")
    for name, m in results.items():
        print(f"{name:16s} {m['steps']:8.1f} {m['completed']:8.2f} {m['collisions']:11.2f} {m['success']:9.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(moving_avg([h["return"] for h in history]))
    ax[0].set(title="Training return (moving avg, 20 ep)", xlabel="episode", ylabel="return")
    names = list(results)
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, [results[n]["steps"] for n in names], 0.4, label="steps to finish")
    ax[1].bar(x + 0.2, [results[n]["collisions"] * 10 for n in names], 0.4, label="collisions x10")
    ax[1].set_xticks(x, names)
    ax[1].set(title="Evaluation (unseen seeds)")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("results/training_and_eval.png", dpi=130)
    print("Saved models/dqn.pt and results/training_and_eval.png")


if __name__ == "__main__":
    main()
