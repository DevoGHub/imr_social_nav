import pickle
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, pkl_file, id_file=None, obs_len=8, pred_len=12):
        self.samples = []
        self.obs_len = obs_len
        self.pred_len = pred_len

        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        # 🔹 filter using split file
        if id_file is not None:
            with open(id_file, "r") as f:
                valid_ids = set(line.strip() for line in f)

            data = {k: v for k, v in data.items() if str(k) in valid_ids}

        # 🔹 create samples
        for traj in data.values():
            positions = [(x, y) for _, x, y in traj]

            if len(positions) < obs_len + pred_len:
                continue

            for i in range(len(positions) - obs_len - pred_len):
                obs = positions[i:i+obs_len]
                fut = positions[i+obs_len:i+obs_len+pred_len]

                obs = torch.tensor(obs, dtype=torch.float32)
                fut = torch.tensor(fut, dtype=torch.float32)

                origin = obs[-1].clone()
                obs = obs - origin
                fut = fut - origin

                self.samples.append((obs, fut))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]