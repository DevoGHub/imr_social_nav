import heapq
import math
from src.sim.costmap import compute_cost


class AStarPlanner:
    def __init__(self, bounds, grid_size=40, alpha=50):
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.grid_size = grid_size
        self.alpha = alpha

    def point_to_grid(self, x, y):
        gx = int((x - self.min_x) / self.grid_size)
        gy = int((y - self.min_y) / self.grid_size)
        return (gx, gy)

    def grid_to_point(self, gx, gy):
        x = self.min_x + gx * self.grid_size
        y = self.min_y + gy * self.grid_size
        return (x, y)

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, node):
        x, y = node
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        return [(x + dx, y + dy) for dx, dy in directions]

    def in_bounds(self, node):
        gx, gy = node
        max_gx = (self.max_x - self.min_x) // self.grid_size
        max_gy = (self.max_y - self.min_y) // self.grid_size

        return (0 <= gx <= max_gx and 0 <= gy <= max_gy)

    def plan(self, start, goal, human_positions, human_histories=None, predictor=None):
        start_node = self.point_to_grid(*start)
        goal_node = self.point_to_grid(*goal)

        open_set = []
        heapq.heappush(open_set, (0, start_node))

        came_from = {}
        g_score = {start_node: 0}

        max_iters = 5000
        iters = 0

        while open_set:
            iters += 1
            if iters > max_iters:
                return []  # early exit (prevents freezing)

            _, current = heapq.heappop(open_set)

            if current == goal_node:
                return self.reconstruct_path(came_from, current, goal)

            for neighbor in self.get_neighbors(current):
                if not self.in_bounds(neighbor):
                    continue

                px, py = self.grid_to_point(*neighbor)

                # cost from humans
                cost = compute_cost(
                    (px, py),
                    human_positions,
                    human_histories,
                    predictor
                )

                tentative_g = g_score[current] + self.grid_size + self.alpha * cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    priority = tentative_g + self.heuristic(neighbor, goal_node)

                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current

        return []  # no path found

    def reconstruct_path(self, came_from, current, goal):
        path = []

        while current in came_from:
            path.append(self.grid_to_point(*current))
            current = came_from[current]

        path.reverse()
        path.append(goal)

        return path