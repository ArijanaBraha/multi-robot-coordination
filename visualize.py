"""Watch the robots.   python visualize.py --policy dqn --seed 3"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from env.env import MultiRobotEnv, STAY

COLORS = [(0, 0, 1), (0, 0.6, 0.6), (0.6, 0, 0.8), (0.9, 0.9, 0)]


def draw(env):
    img = np.ones((env.size, env.size, 3)) * 0.88
    img[env.grid == 1] = (0.8, 0, 0)
    for t in env.tasks:
        if t.arrived and not t.done:
            img[t.location] = (0, 0.8, 0)
    for o in env.obstacles:
        img[o["pos"]] = (1, 0.55, 0)
    for r in env.robots:
        img[r.pos] = COLORS[r.robot_id % len(COLORS)]
    plt.clf()
    plt.imshow(img)
    plt.title(f"step {env.steps}  done {env.completed}/{env.n_tasks}  collisions {env.collisions}")
    plt.xticks([]); plt.yticks([])
    plt.pause(0.15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["astar", "dqn"], default="dqn")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--robots", type=int, default=2)
    args = p.parse_args()

    env = MultiRobotEnv(n_robots=args.robots)
    obs = env.reset(seed=args.seed)
    agent = None
    if args.policy == "dqn":
        from agents.agents import DQNAgent
        agent = DQNAgent(env.obs_dim, env.n_actions)
        agent.load("models/dqn.pt")

    plt.ion()
    while True:
        acts = []
        for i, o in enumerate(obs):
            r = env.robots[i]
            if r.task is None:
                acts.append(STAY)
            elif agent is None:
                acts.append(env.hint_action(r))
            else:
                acts.append(agent.select_action(o, greedy=True))
        obs, _, term, trunc, _ = env.step(acts)
        draw(env)
        if term or trunc:
            break
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
