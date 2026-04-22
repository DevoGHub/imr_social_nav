import random
import math
from src.sim.collision import is_collision
from src.sim.costmap import compute_cost

class Human:
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.index = 0

    def step(self):
        if self.index < len(self.trajectory) - 1:
            self.index += 1

    def get_position(self):
        _, x, y = self.trajectory[self.index]
        return x, y


class Robot:
    def __init__(self, start, goal, bounds, step_size=5.0):
        self.pos = list(start)
        self.goal = goal
        self.step_size = step_size
        self.bounds = bounds
        self.prev_dist = float("inf")
        self.stuck_steps = 0    

    def step(self, human_positions):
        # candidate directions (8 directions)
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        best_pos = None
        best_score = float("inf")

        for dx, dy in directions:
            # normalize direction
            norm = math.sqrt(dx**2 + dy**2)
            dx /= norm
            dy /= norm

            new_x = self.pos[0] + dx * self.step_size
            new_y = self.pos[1] + dy * self.step_size

            # goal distance
            goal_dist = math.sqrt((self.goal[0] - new_x)**2 + (self.goal[1] - new_y)**2)

            # social cost
            cost = compute_cost((new_x, new_y), human_positions)

            # combined score
            score = goal_dist + 50 * cost

            min_x, max_x, min_y, max_y = self.bounds

            if not (min_x <= new_x <= max_x and min_y <= new_y <= max_y):
                continue    

            if score < best_score:
                best_score = score
                best_pos = (new_x, new_y)

        if best_pos:
            self.pos = list(best_pos)

        current_dist = math.sqrt(
            (self.goal[0] - self.pos[0])**2 +
            (self.goal[1] - self.pos[1])**2
        )

        # check progress
        if abs(current_dist - self.prev_dist) < 0.5:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        self.prev_dist = current_dist

    def reached_goal(self):
        dx = self.goal[0] - self.pos[0]
        dy = self.goal[1] - self.pos[1]
        return (dx**2 + dy**2) ** 0.5 < self.step_size
    
    def is_stuck(self, threshold=20):
        return self.stuck_steps > threshold


class Environment:
    def __init__(self, trajectories):
        self.humans = [Human(traj) for traj in trajectories.values()]
        # self.robot = Robot(start=(500, 500), goal=(1000, 1000))
        self.time = 0
        start = self.sample_safe_point(trajectories)
        self.goal = self.sample_point_from_trajectories(trajectories)
        while self.goal == start:
            self.goal = self.sample_point_from_trajectories(trajectories)

        self.min_x, self.max_x, self.min_y, self.max_y = self.compute_bounds(trajectories)
        self.robot = Robot(start=start, goal=self.goal, bounds=(self.min_x, self.max_x, self.min_y, self.max_y))


    def step(self):
        for human in self.humans:
            human.step()

        human_positions = [h.get_position() for h in self.humans]

        if not is_collision(self.robot.pos, human_positions):
            self.robot.step(human_positions)

        self.time += 1

    def get_state(self):
        return {
            "humans": [h.get_position() for h in self.humans],
            "robot": self.robot.pos,
            "goal": self.goal
        }
    
    def sample_point_from_trajectories(self, trajectories):
        traj = random.choice(list(trajectories.values()))
        _, x, y = random.choice(traj)
        return (x, y)
    
    def sample_safe_point(self, trajectories, min_dist=30, max_tries=100):
        human_positions = [h.get_position() for h in self.humans]

        for _ in range(max_tries):
            traj = random.choice(list(trajectories.values()))
            _, x, y = random.choice(traj)

            # check distance constraint
            too_close = False
            for hx, hy in human_positions:
                if ((x - hx)**2 + (y - hy)**2)**0.5 < min_dist:
                    too_close = True
                    break

            if too_close:
                continue

            # check cost constraint
            cost = compute_cost((x, y), human_positions)

            if cost < 0.1:  # threshold (tune later)
                return (x, y)

        # fallback (if nothing found)
        return (x, y)
    
    def compute_bounds(self, trajectories):
        xs, ys = [], []

        for traj in trajectories.values():
            for _, x, y in traj:
                xs.append(x)
                ys.append(y)

        return min(xs), max(xs), min(ys), max(ys)