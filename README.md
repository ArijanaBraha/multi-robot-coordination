# Multi-Robot Coordination: DQN + A* + auction allocation

Grid world with walls, moving obstacles and tasks that arrive over time.

## What is implemented (and only this)
- **RL:** Double DQN, one network shared by all robots (parameter sharing). The RL action directly moves the robot.
- **Planning:** A* with dynamic obstacles treated as blocked cells. Returns `None` if the goal is unreachable.
  The next A* step is given to the network as a one-hot *hint* (hybrid classical planning + RL).
- **Allocation:** sequential auction. Tasks are auctioned by priority, only idle robots bid, bid = A* path length + fairness term.
- **Not implemented:** MADQL / PPO / RRT*. (Possible future work.)

## Run
    pip install -r requirements.txt
    python train.py --episodes 600 --eval-episodes 100   # trains, saves models/dqn.pt, compares with A*-only
    python visualize.py --policy dqn --seed 3            # or --policy astar

## Evaluation
`train.py` compares "A* only" and "A* hint + DQN" on unseen seeds: steps to finish, tasks completed,
collisions, fraction of episodes where all tasks were completed. Report whatever numbers you actually get.
