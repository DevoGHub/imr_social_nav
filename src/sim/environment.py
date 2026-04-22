import random
import math

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
    def __init__(self, start, goal, step_size=5.0):
        self.pos = list(start)
        self.goal = goal
        self.step_size = step_size

    def step(self):
        dx = self.goal[0] - self.pos[0]
        dy = self.goal[1] - self.pos[1]

        dist = math.sqrt(dx**2 + dy**2)

        # stop if reached
        if dist < self.step_size:
            return

        # normalize
        dx /= dist
        dy /= dist

        # move
        self.pos[0] += dx * self.step_size
        self.pos[1] += dy * self.step_size


class Environment:
    def __init__(self, trajectories):
        self.humans = [Human(traj) for traj in trajectories.values()]
        # self.robot = Robot(start=(500, 500), goal=(1000, 1000))
        self.time = 0
        start = self.sample_point_from_trajectories(trajectories)
        self.goal = self.sample_point_from_trajectories(trajectories)
        while self.goal == start:
            self.goal = self.sample_point_from_trajectories(trajectories)

        self.robot = Robot(start=start, goal=self.goal)

    def step(self):
        for human in self.humans:
            human.step()

        self.robot.step()
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