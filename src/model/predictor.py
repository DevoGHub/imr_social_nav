import torch
from src.model.lstm import TrajectoryLSTM


class Predictor:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TrajectoryLSTM().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, history):
        """
        history: list of (x, y) of length obs_len (8)
        returns: list of (x, y) of length pred_len (12)
        """

        obs = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 🔹 normalize (same as training)
        origin = obs[:, -1:, :].clone()
        obs = obs - origin

        with torch.no_grad():
            pred = self.model(obs)  # (1, pred_len, 2)

        # 🔹 de-normalize back to world coordinates
        pred = pred + origin

        return pred.squeeze(0).cpu().numpy()