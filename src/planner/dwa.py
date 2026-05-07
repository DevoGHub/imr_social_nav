import math
import numpy as np
from src.sim.costmap import compute_cost


class DWAPlanner:
    def __init__(
        self,
        max_speed=5.0,       # matches robot step_size
        min_speed=0.0,
        max_yaw_rate=math.pi / 4,   # step
        max_accel=3.0,
        max_delta_yaw_rate=math.pi / 8,
        v_resolution=0.5,
        yaw_rate_resolution=0.1,
        # scoring weights
        w_heading=0.8,
        w_clearance=1.5,
        w_velocity=0.2,
        use_prediction=False,
    ):
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_yaw_rate = max_yaw_rate
        self.max_accel = max_accel
        self.max_delta_yaw_rate = max_delta_yaw_rate
        self.v_resolution = v_resolution
        self.yaw_rate_resolution = yaw_rate_resolution

        self.w_heading = w_heading
        self.w_clearance = w_clearance
        self.w_velocity = w_velocity

        self.use_prediction = use_prediction

    def _dynamic_window(self, v, w):
        v_min = max(self.min_speed,   v - self.max_accel)
        v_max = min(self.max_speed,   v + self.max_accel)
        w_min = max(-self.max_yaw_rate, w - self.max_delta_yaw_rate)
        w_max = min( self.max_yaw_rate, w + self.max_delta_yaw_rate)
        return v_min, v_max, w_min, w_max

    def _simulate(self, x, y, yaw, v, w, dt=1.0):
        new_yaw = yaw + w * dt
        new_x   = x + v * math.cos(new_yaw) * dt
        new_y   = y + v * math.sin(new_yaw) * dt
        return new_x, new_y, new_yaw

    def _heading_score(self, x, y, yaw, goal):
        goal_angle = math.atan2(goal[1] - y, goal[0] - x)
        angle_diff = abs(goal_angle - yaw)
        # normalise to [0, pi]
        while angle_diff > math.pi:
            angle_diff = abs(angle_diff - 2 * math.pi)
        return (math.pi - angle_diff) / math.pi   # 1 = perfectly aligned

    def _clearance_score(self, x, y, human_positions,
                         human_histories=None, predictor=None):
        cost = self._get_cost(
            (x, y), human_positions, human_histories, predictor
        )
        return 1.0 / (1.0 + cost)

    def _get_cost(self, point, human_positions,
                  human_histories=None, predictor=None):
        use_pred = self.use_prediction and predictor is not None and human_histories is not None
        return compute_cost(
            point,
            human_positions,
            human_histories if use_pred else None,
            predictor       if use_pred else None,
        )

    def plan(self, robot_pos, robot_vel, robot_yaw, goal,
             human_positions, human_histories=None, predictor=None):
        x, y = robot_pos
        vx, vy = robot_vel
        v_cur = math.sqrt(vx**2 + vy**2)   # current speed
        w_cur = 0.0                          # assume zero yaw rate at start

        v_min, v_max, w_min, w_max = self._dynamic_window(v_cur, w_cur)

        best_score = -float("inf")
        best_v = v_min
        best_w = 0.0

        v_samples = np.arange(v_min, v_max + self.v_resolution, self.v_resolution)
        w_samples = np.arange(w_min, w_max + self.yaw_rate_resolution,
                              self.yaw_rate_resolution)

        for v in v_samples:
            for w in w_samples:
                nx, ny, nyaw = self._simulate(x, y, robot_yaw, v, w)

                h = self._heading_score(nx, ny, nyaw, goal)
                c = self._clearance_score(
                    nx, ny, human_positions, human_histories, predictor
                )
                s = v / self.max_speed   # velocity score

                score = (self.w_heading   * h +
                         self.w_clearance * c +
                         self.w_velocity  * s)

                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w

        return best_v, best_w

    def next_position(self, robot_pos, robot_vel, robot_yaw, goal,
                      human_positions, human_histories=None, predictor=None):
        best_v, best_w = self.plan(
            robot_pos, robot_vel, robot_yaw, goal,
            human_positions, human_histories, predictor
        )
        nx, ny, nyaw = self._simulate(*robot_pos, robot_yaw, best_v, best_w)
        new_vx = best_v * math.cos(nyaw)
        new_vy = best_v * math.sin(nyaw)
        return nx, ny, nyaw, new_vx, new_vy
