import pickle
import random
import numpy as np
import time
from src.sim.environment import Environment
from src.sim.visualizer import Visualizer

with open("data/processed/bookstore_video0_traj.pkl", "rb") as f:
    traj = pickle.load(f)


def run_episode(traj, mode, max_steps=1000, render=False):
    env = Environment(traj)
    vis = Visualizer() if render else None

    success = False

    start_time = time.time()

    for _ in range(max_steps):
        env.step(mode=mode)

        if render:
            vis.render(env.get_state())

        if env.robot.reached_goal():
            success = True
            break

    runtime = time.time() - start_time

    return {
        "success": success,
        "collisions": env.collision_count,
        "min_dist": env.min_dist,
        "path_length": env.path_length,
        "steps": env.steps_taken,
        "runtime": runtime,
    }


if __name__ == "__main__":
    modes = ["greedy", "astar", "lstm"]
    num_runs = 5
    max_steps = 1000

    results = {mode: [] for mode in modes}

    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}")

        for mode in modes:
            seed = i
            random.seed(seed)
            np.random.seed(seed)

            res = run_episode(traj, mode, max_steps=max_steps, render=False)
            results[mode].append(res)

            print(
                f"  {mode:7} | success={res['success']} | "
                f"coll={res['collisions']} | min_dist={res['min_dist']:.2f}"
            )

    def summarize(results):
        n = len(results)
        successful = [r for r in results if r["success"]]

        return {
            "success_rate": sum(r["success"] for r in results) / n,
            "avg_collisions": sum(r["collisions"] for r in results) / n,
            "avg_collisions_success": (
                sum(r["collisions"] for r in successful) / len(successful)
                if successful else float("nan")
            ),
            "avg_min_dist": sum(r["min_dist"] for r in results) / n,
            "median_min_dist": sorted(r["min_dist"] for r in results)[n // 2],
            "avg_path_len": (
                sum(r["path_length"] for r in successful) / len(successful)
                if successful else float("nan")
            ),
            "avg_steps_success": (
                sum(r["steps"] for r in successful) / len(successful)
                if successful else float("nan")
            ),
            "avg_runtime": sum(r["runtime"] for r in results) / n,
        }

    for mode in modes:
        stats = summarize(results[mode])

        print(f"\n=== {mode.upper()} ===")

        for k, v in stats.items():
            if isinstance(v, float):
                print(f"{k:25}: {v:.4f}")
            else:
                print(f"{k:25}: {v}")