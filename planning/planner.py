import heapq


class DynamicPlanner:
    """A* on a 4-connected grid. grid: numpy array, 1 = static wall.

    `blocked` is an optional set of extra cells (e.g. moving obstacles) that
    are treated as walls for this one query. This is what makes planning
    "dynamic": the grid stays static, the blocked set changes every step.
    """

    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape

    def in_bounds(self, node):
        return 0 <= node[0] < self.rows and 0 <= node[1] < self.cols

    def is_free(self, node, blocked=None):
        return (self.in_bounds(node)
                and self.grid[node[0], node[1]] == 0
                and (not blocked or node not in blocked))

    @staticmethod
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal, blocked=None):
        """Returns list of cells from the first step to the goal (start excluded),
        [] if start == goal, or None if the goal is unreachable."""
        start, goal = tuple(start), tuple(goal)
        if start == goal:
            return []
        blocked = set(blocked) - {start, goal} if blocked else set()
        if not self.is_free(goal):
            return None

        open_heap = [(self.heuristic(start, goal), 0, start)]
        came_from = {}
        g_score = {start: 0}
        closed = set()
        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current == goal:
                path = []
                while current != start:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            if current in closed:
                continue
            closed.add(current)
            for dx, dy in self.MOVES:
                n = (current[0] + dx, current[1] + dy)
                if not self.is_free(n, blocked):
                    continue
                ng = g + 1
                if ng < g_score.get(n, float("inf")):
                    g_score[n] = ng
                    came_from[n] = current
                    heapq.heappush(open_heap, (ng + self.heuristic(n, goal), ng, n))
        return None  # no route (NOT a straight line through walls)

    def _path_valid(self, path, start, goal, blocked):
        if not path or path[-1] != goal:
            return False
        if self.heuristic(path[0], start) != 1:
            return False
        blocked = set(blocked) - {goal} if blocked else None
        return all(self.is_free(n, blocked) for n in path)

    def replan_if_blocked(self, path, start, goal, blocked=None):
        """Keep the old path if it is still valid, otherwise plan again.
        If no route exists while dynamic obstacles are considered, fall back to
        the static route (obstacles move away, so waiting is better than giving up)."""
        start, goal = tuple(start), tuple(goal)
        if path is not None and self._path_valid(path, start, goal, blocked):
            return path
        new_path = self.astar(start, goal, blocked)
        if new_path is None and blocked:
            new_path = self.astar(start, goal, None)
        return new_path
