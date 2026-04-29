import random
import math
from src.sim.costmap import compute_cost
from src.planner.astar import AStarPlanner
from src.model.predictor import Predictor


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
    
    def get_history(self, obs_len=8):
        start = max(0, self.index - obs_len + 1)
        traj_slice = self.trajectory[start:self.index + 1]

        if len(traj_slice) < obs_len:
            traj_slice = [traj_slice[0]] * (obs_len - len(traj_slice)) + traj_slice

        return [(x, y) for _, x, y in traj_slice]


class Robot:
    def __init__(self, start, goal, bounds, step_size=5.0):
        self.pos = list(start)
        self.goal = goal
        self.step_size = step_size
        self.bounds = bounds
        self.prev_dist = float("inf")
        self.stuck_steps = 0

    def reached_goal(self, tol=10):
        dx = self.goal[0] - self.pos[0]
        dy = self.goal[1] - self.pos[1]
        return (dx**2 + dy**2)**0.5 < tol

    def is_stuck(self, threshold=20):
        return self.stuck_steps > threshold


class Environment:
    def __init__(self, trajectories):
        self.humans = [Human(traj) for traj in trajectories.values()]
        self.time = 0

        self.min_x, self.max_x, self.min_y, self.max_y = self.compute_bounds(trajectories)

        start = self.sample_safe_point(trajectories)
        self.goal = self.sample_point_from_trajectories(trajectories)

        while self.goal == start:
            self.goal = self.sample_point_from_trajectories(trajectories)

        self.robot = Robot(start=start, goal=self.goal,
                           bounds=(self.min_x, self.max_x, self.min_y, self.max_y))

        self.planner = AStarPlanner(
            bounds=(self.min_x, self.max_x, self.min_y, self.max_y)
        )

        self.path = []
        self.path_index = 0

        # metrics
        self.collision_count = 0
        self.min_dist = float("inf")
        self.path_length = 0
        self.steps_taken = 0

        # cost weight
        self.alpha = 100

        self.predictor = Predictor("models/trajectory_lstm.pth")

    def step(self, mode="astar"):
        prev_pos = self.robot.pos.copy()

        if self.predictor is not None:
            self.predictor.reset_cache()

        # move humans
        for human in self.humans:
            human.step()

        human_positions = [h.get_position() for h in self.humans]
        human_histories = [h.get_history() for h in self.humans]

        self.predicted_trajs = None

        if mode == "lstm" and self.predictor is not None:
            self.predicted_trajs = [
                self.predictor.predict(hist)
                for hist in human_histories
            ]

        if self.predictor is not None:
            self.predictor.precomputed = self.predicted_trajs

        # choose behavior
        if mode == "greedy":
            self.greedy_step(human_positions)

        elif mode == "astar":
            self.astar_step(human_positions)

        elif mode == "lstm":
            self.lstm_step(human_positions, human_histories)

        # time update
        self.time += 1
        self.steps_taken = self.time

        # metrics
        if self.check_collision(self.robot.pos, human_positions):
            self.collision_count += 1

        for hx, hy in human_positions:
            dist = ((self.robot.pos[0]-hx)**2 + (self.robot.pos[1]-hy)**2)**0.5
            self.min_dist = min(self.min_dist, dist)

        dx = self.robot.pos[0] - prev_pos[0]
        dy = self.robot.pos[1] - prev_pos[1]
        self.path_length += (dx**2 + dy**2)**0.5

    def greedy_step(self, human_positions, human_histories=None):
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        best_pos = None
        best_score = float("inf")

        for dx, dy in directions:
            norm = math.sqrt(dx**2 + dy**2)
            dx /= norm
            dy /= norm

            new_x = self.robot.pos[0] + dx * self.robot.step_size
            new_y = self.robot.pos[1] + dy * self.robot.step_size

            if not (self.min_x <= new_x <= self.max_x and
                    self.min_y <= new_y <= self.max_y):
                continue

            goal_dist = math.sqrt(
                (self.goal[0] - new_x)**2 +
                (self.goal[1] - new_y)**2
            )

            cost = compute_cost(
                (new_x, new_y),
                human_positions,
                human_histories,
                self.predictor
            )

            score = goal_dist + self.alpha * cost

            if score < best_score:
                best_score = score
                best_pos = (new_x, new_y)

        if best_pos:
            self.robot.pos = list(best_pos)

    def astar_step(self, human_positions, human_histories=None):
        if self.time % 20 == 0 or self.path_index >= len(self.path):
            self.path = self.planner.plan(
                self.robot.pos,
                self.goal,
                human_positions,
                human_histories,
                self.predictor
            )
            self.path_index = 0

        if self.path and self.path_index < len(self.path):
            next_point = self.path[self.path_index]

            dx = next_point[0] - self.robot.pos[0]
            dy = next_point[1] - self.robot.pos[1]

            dist = (dx**2 + dy**2)**0.5

            if dist < self.robot.step_size:
                self.path_index += 1
            else:
                self.robot.pos[0] += dx / dist * self.robot.step_size
                self.robot.pos[1] += dy / dist * self.robot.step_size

    def lstm_step(self, human_positions, human_histories):
        self.astar_step(human_positions, human_histories)

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

            too_close = False
            for hx, hy in human_positions:
                if ((x - hx)**2 + (y - hy)**2)**0.5 < min_dist:
                    too_close = True
                    break

            if too_close:
                continue

            cost = compute_cost((x, y), human_positions, None, None)

            if cost < 0.1:
                return (x, y)

        return (x, y)

    def compute_bounds(self, trajectories):
        xs, ys = [], []

        for traj in trajectories.values():
            for _, x, y in traj:
                xs.append(x)
                ys.append(y)

        return min(xs), max(xs), min(ys), max(ys)

    def check_collision(self, robot_pos, humans, threshold=10):
        for hx, hy in humans:
            if ((robot_pos[0]-hx)**2 + (robot_pos[1]-hy)**2)**0.5 < threshold:
                return True
        return False