class Task:
    def __init__(self, task_id, location, priority=1, arrival_step=0):
        self.task_id = task_id
        self.location = tuple(location)
        self.priority = priority
        self.arrival_step = arrival_step   # step at which the task appears
        self.arrived = arrival_step == 0
        self.assigned = False
        self.done = False


class AuctionBasedAllocator:
    """Sequential single-item auction.

    * Tasks are auctioned in order of priority (high first).
    * Only IDLE robots bid, so a robot keeps its task until it finishes it
      (no re-assignment every step -> no oscillation).
    * Bid = path length to the task + fairness * tasks already completed.
      Lowest bid wins.
    The weights are constructor parameters, not hidden constants.
    """

    def __init__(self, robots, planner, fairness=0.5):
        self.robots = robots
        self.planner = planner
        self.fairness = fairness

    def bid(self, robot, task):
        path = self.planner.astar(robot.pos, task.location)
        if path is None:
            return float("inf")
        return len(path) + self.fairness * robot.completed

    def allocate(self, tasks):
        idle = [r for r in self.robots if r.task is None]
        pending = [t for t in tasks if t.arrived and not t.assigned and not t.done]
        pending.sort(key=lambda t: -t.priority)
        assignments = {}
        for task in pending:
            if not idle:
                break
            bids = [(self.bid(r, task), r.robot_id, r) for r in idle]
            best_cost, _, winner = min(bids, key=lambda b: (b[0], b[1]))
            if best_cost == float("inf"):
                continue
            winner.task = task
            winner.path = None
            task.assigned = True
            idle.remove(winner)
            assignments[task.task_id] = winner.robot_id
        return assignments
