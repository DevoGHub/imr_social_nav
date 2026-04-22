import pickle


def create_sequences(traj, obs_len=8, pred_len=12):
    sequences = []

    if len(traj) < obs_len + pred_len:
        return sequences

    for i in range(len(traj) - obs_len - pred_len + 1):
        obs = traj[i:i + obs_len]
        pred = traj[i + obs_len:i + obs_len + pred_len]

        obs_xy = [(x, y) for _, x, y in obs]
        pred_xy = [(x, y) for _, x, y in pred]

        sequences.append({
            "obs": obs_xy,
            "pred": pred_xy
        })

    return sequences


def process_all_trajectories(trajectories, obs_len=8, pred_len=12):
    all_sequences = []

    for traj in trajectories.values():
        seqs = create_sequences(traj, obs_len, pred_len)
        all_sequences.extend(seqs)

    return all_sequences


def normalize_sequences(sequences):
    normalized = []

    for seq in sequences:
        obs = seq["obs"]
        pred = seq["pred"]

        # Relative Coordinates used for LSTM
        origin_x, origin_y = obs[0]

        obs_rel = [(x - origin_x, y - origin_y) for x, y in obs]
        pred_rel = [(x - origin_x, y - origin_y) for x, y in pred]

        normalized.append({
            "obs": obs_rel,
            "pred": pred_rel
        })

    return normalized

# For LSTM
def save_processed_data(sequences, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(sequences, f)

# For Sim
def save_trajectories(trajectories, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)