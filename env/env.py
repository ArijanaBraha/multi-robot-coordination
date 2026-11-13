import random
from collections import deque

import numpy as np

from planning.planner import DynamicPlanner
from allocation.task_allocation import Task, AuctionBasedAllocator

# up, down, left, right, stay  (same indexing everywhere)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
STAY = 4
VIEW = 2  # local view radius -> 5x5


class Robot:
    def __init__(self, robot_id, pos):
        self.robot_id = robot_id
        self.pos = tuple(pos)
        self.task = None
        self.path = None
        self.completed = 0


class MultiRobotEnv:
    """Grid world: static walls, moving obstacles, tasks that arrive over time.

    The RL action DIRECTLY moves the robot. A* is used as a *hint* in the
    observation (one-hot of its next move) and for task allocation costs, so
    the network learns when to follow the planner and when to deviate.
    """

    WALLS = [(2, 3), (3, 2), (4, 5)]
    OBSTACLE_START = [((1, 4), (0, 1)), ((5, 1), (1, 0))]
    FIXED_TASKS = [((6, 6), 2), ((5, 7), 1), ((7, 2), 3), ((6, 3), 1), ((2, 6), 2)]

    def __init__(self, size=8, n_robots=2, max_steps=200, n_tasks=5,
                 n_initial_tasks=3, n_dynamic_obstacles=2,
                 randomize_tasks=True, seed=None):
        assert n_robots <= size and n_initial_tasks <= n_tasks
        self.size = size
        self.n_robots = n_robots
        self.max_steps = max_steps
        self.n_tasks = n_tasks
        self.n_initial_tasks = n_initial_tasks
        self.n_dynamic_obstacles = n_dynamic_obstacles
        self.randomize_tasks = randomize_tasks
        self.rng = random.Random(seed)

        self.grid = np.zeros((size, size), dtype=int)
        for w in self.WALLS:
            self.grid[w] = 1
        self.planner = DynamicPlanner(self.grid)

        self.n_actions = len(ACTIONS)
        side = 2 * VIEW + 1
        # own pos(2) + goal delta(2) + 3 view channels + A* hint(5) + has_task(1)
        self.obs_dim = 2 + 2 + 3 * side * side + self.n_actions + 1
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.steps = 0
        self.collisions = 0
        self.completed = 0
        self.robots = [Robot(i, (0, i)) for i in range(self.n_robots)]
        starts = {r.pos for r in self.robots}
        free = [(x, y) for x in range(self.size) for y in range(self.size)
                if self.grid[x, y] == 0 and (x, y) not in starts]

        if self.randomize_tasks:
            locs = self.rng.sample(free, self.n_tasks)
            prios = [self.rng.choice([1, 2, 3]) for _ in locs]
        else:
            locs = [l for l, _ in self.FIXED_TASKS[:self.n_tasks]]
            prios = [p for _, p in self.FIXED_TASKS[:self.n_tasks]]
        n_late = self.n_tasks - self.n_initial_tasks
        arrivals = [0] * self.n_initial_tasks + sorted(self.rng.sample(range(5, 60), n_late))
        self.tasks = [Task(100 + i, locs[i], prios[i], arrivals[i]) for i in range(self.n_tasks)]
        self._dist_maps = {t.task_id: self._bfs(t.location) for t in self.tasks}

        self.obstacles = [{"pos": p, "dir": d}
                          for p, d in self.OBSTACLE_START[:self.n_dynamic_obstacles]]
        self.allocator = AuctionBasedAllocator(self.robots, self.planner)
        self._allocate()
        return self._observe_all()

    def _bfs(self, goal):
        dist = np.full((self.size, self.size), 10 ** 6, dtype=int)
        dist[goal] = 0
        q = deque([goal])
        while q:
            x, y = q.popleft()
            for dx, dy in DynamicPlanner.MOVES:
                n = (x + dx, y + dy)
                if self.planner.is_free(n) and dist[n] > dist[x, y] + 1:
                    dist[n] = dist[x, y] + 1
                    q.append(n)
        return dist

    def _dist(self, robot):
        return 0 if robot.task is None else int(self._dist_maps[robot.task.task_id][robot.pos])

    # -------------------------------------------------------- moving obstacles
    def _obstacle_next(self, o):
        x, y = o["pos"]
        dx, dy = o["dir"]
        n = (x + dx, y + dy)
        if self.planner.is_free(n):
            return n, o["dir"]
        d = (-dx, -dy)
        n = (x + d[0], y + d[1])
        if self.planner.is_free(n):
            return n, d
        return (x, y), d

    def _blocked_cells(self):
        cells = set()
        for o in self.obstacles:
            cells.add(o["pos"])
            cells.add(self._obstacle_next(o)[0])
        return cells

    def _move_obstacles(self):
        for o in self.obstacles:
            o["pos"], o["dir"] = self._obstacle_next(o)

    # ------------------------------------------------------------ planner hint
    def _update_path(self, r):
        if r.task is None:
            r.path = None
            return
        r.path = self.planner.replan_if_blocked(r.path, r.pos, r.task.location,
                                                self._blocked_cells())

    def hint_action(self, r):
        """Action that follows the current A* path (STAY if none)."""
        self._update_path(r)
        if not r.path:
            return STAY
        nx, ny = r.path[0]
        return ACTIONS.index((nx - r.pos[0], ny - r.pos[1]))

    # ------------------------------------------------------------ observations
    def _observe(self, r):
        s = self.size - 1
        rx, ry = r.pos
        if r.task is not None:
            gx, gy = r.task.location
            delta = [(gx - rx) / s, (gy - ry) / s]
        else:
            delta = [0.0, 0.0]

        cur = {o["pos"] for o in self.obstacles}
        nxt = {self._obstacle_next(o)[0] for o in self.obstacles}
        others = {q.pos for q in self.robots if q.robot_id != r.robot_id}
        side = 2 * VIEW + 1
        walls = np.zeros((side, side))
        dyn = np.zeros((side, side))
        oth = np.zeros((side, side))
        for i in range(-VIEW, VIEW + 1):
            for j in range(-VIEW, VIEW + 1):
                c = (rx + i, ry + j)
                a, b = i + VIEW, j + VIEW
                if not self.planner.is_free(c):
                    walls[a, b] = 1          # walls and out-of-bounds
                    continue
                if c in cur:
                    dyn[a, b] = 1.0
                elif c in nxt:
                    dyn[a, b] = 0.5          # where an obstacle is heading
                if c in others:
                    oth[a, b] = 1.0

        hint = np.zeros(self.n_actions)
        if r.task is not None:
            hint[self.hint_action(r)] = 1.0
        return np.concatenate([
            [rx / s, ry / s], delta,
            walls.ravel(), dyn.ravel(), oth.ravel(),
            hint, [1.0 if r.task is not None else 0.0],
        ]).astype(np.float32)

    def _observe_all(self):
        return [self._observe(r) for r in self.robots]

    # -------------------------------------------------------------------- step
    def _allocate(self):
        self.allocator.allocate(self.tasks)

    def step(self, actions):
        """actions: one int per robot (ignored for idle robots).
        Returns obs, rewards, terminated, truncated, info."""
        self.steps += 1
        n = self.n_robots
        rewards = [0.0] * n
        active = [r.task is not None for r in self.robots]
        prev_d = [self._dist(r) for r in self.robots]
        obs_now = {o["pos"] for o in self.obstacles}

        proposed = []
        for i, r in enumerate(self.robots):
            if not active[i]:
                proposed.append(r.pos)
                continue
            rewards[i] -= 0.05  # time cost
            dx, dy = ACTIONS[int(actions[i])]
            cell = (r.pos[0] + dx, r.pos[1] + dy)
            if not self.planner.is_free(cell):
                rewards[i] -= 1.0          # bumped into wall / border
                proposed.append(r.pos)
            elif cell in obs_now:
                rewards[i] -= 10.0         # tried to enter a moving obstacle
                self.collisions += 1
                proposed.append(r.pos)
            else:
                proposed.append(cell)

        # robot-robot conflicts: same target cell, swaps, or entering an occupied cell
        changed = True
        while changed:
            changed = False
            for i, r in enumerate(self.robots):
                if proposed[i] == r.pos:
                    continue
                for j, q in enumerate(self.robots):
                    if j == i:
                        continue
                    if proposed[j] == proposed[i] or (proposed[j] == r.pos and proposed[i] == q.pos):
                        proposed[i] = r.pos
                        rewards[i] -= 5.0
                        self.collisions += 1
                        changed = True
                        break

        for i, r in enumerate(self.robots):
            r.pos = proposed[i]
            if r.path and r.path[0] == r.pos:
                r.path = r.path[1:]

        self._move_obstacles()
        new_obs = {o["pos"] for o in self.obstacles}
        for i, r in enumerate(self.robots):
            if r.pos in new_obs:           # obstacle drove into the robot
                rewards[i] -= 10.0
                self.collisions += 1

        for t in self.tasks:
            if not t.arrived and t.arrival_step <= self.steps:
                t.arrived = True

        for i, r in enumerate(self.robots):
            if not active[i]:
                continue
            if r.pos == r.task.location:
                rewards[i] += 10.0
                r.task.done = True
                r.task = None
                r.path = None
                r.completed += 1
                self.completed += 1
            else:
                rewards[i] += 0.3 * (prev_d[i] - self._dist(r))   # progress shaping

        self._allocate()
        terminated = all(t.done for t in self.tasks)
        truncated = (not terminated) and self.steps >= self.max_steps
        info = {"active": active, "collisions": self.collisions,
                "completed": self.completed, "steps": self.steps}
        return self._observe_all(), rewards, terminated, truncated, info
