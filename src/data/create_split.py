import pickle
import random
import os

def create_split(pkl_path, output_dir, train_ratio=0.7, seed=42):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    keys = list(data.keys())
    random.seed(seed)
    random.shuffle(keys)

    split_idx = int(len(keys) * train_ratio)

    train_keys = keys[:split_idx]
    test_keys = keys[split_idx:]

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "train.txt"), "w") as f:
        for k in train_keys:
            f.write(f"{k}\n")

    with open(os.path.join(output_dir, "test.txt"), "w") as f:
        for k in test_keys:
            f.write(f"{k}\n")

    print(f"Split created at {output_dir}")


if __name__ == "__main__":
    create_split(
        "data/processed/bookstore_video0_traj.pkl",
        "data/splits/bookstore_video0"
    )