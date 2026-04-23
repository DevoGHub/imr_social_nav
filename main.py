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

'''LST Dataset check'''
# from src.model.dataset import TrajectoryDataset

# files = [
#     "data/processed/bookstore_video0_traj.pkl",
#     "data/processed/bookstore_video1_traj.pkl",
#     "data/processed/coupa_video0_traj.pkl",
# ]

# dataset = TrajectoryDataset(files)

# print(len(dataset))
# obs, fut = dataset[0]

# print(obs.shape)  # should be (8, 2)
# print(fut.shape)  # should be (12, 2)

'''LSTM check'''
# from src.model.lstm import TrajectoryLSTM
# import torch

# model = TrajectoryLSTM()

# dummy = torch.randn(4, 8, 2)  # batch of 4
# out = model(dummy)

# print(out.shape)  # should be (4, 12, 2)

'''Visualisation from pickles'''
# import pickle
# from src.experiment.main_experiment import run_episode

# with open("data/processed/bookstore_video0_traj.pkl", "rb") as f:
#     traj = pickle.load(f)

# mode = "greedy"

# res = run_episode(traj, mode=mode, render=True)

# print("\n=== RESULT ===")
# for k, v in res.items():
#     print(f"{k}: {v}")