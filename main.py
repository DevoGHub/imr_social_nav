'''Prepoc Test'''
# from src.data.parser import build_trajectories, clean_trajectories

# traj = build_trajectories("data/raw/coupa/video0.txt")
# traj = clean_trajectories(traj)

# print(len(traj))
# print(list(traj.items())[0])


'''Create data pickles'''
# from src.data.parser import build_trajectories, clean_trajectories
# from src.data.preprocess import process_all_trajectories, normalize_sequences, save_processed_data, save_trajectories

# traj = build_trajectories("data/raw/coupa/video0.txt")
# traj = clean_trajectories(traj)

# sequences = process_all_trajectories(traj)
# sequences = normalize_sequences(sequences)

# print(len(sequences))
# print(sequences[0])

# save_processed_data(sequences, "data/processed/coupa_video0.pkl")
# save_trajectories(traj, "data/processed/coupa_video0_traj.pkl")

'''Visualisation from pickles'''
import pickle
from src.experiment.main_experiment import run_episode

with open("data/processed/bookstore_video0_traj.pkl", "rb") as f:
    traj = pickle.load(f)

mode = "astar"

res = run_episode(traj, mode=mode, render=True)

print("\n=== RESULT ===")
for k, v in res.items():
    print(f"{k}: {v}")