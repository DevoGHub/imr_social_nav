import torch
from src.model.lstm import TrajectoryLSTM


class Predictor:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TrajectoryLSTM().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        # cache
        self.cache = {}

    def reset_cache(self):
        self.cache = {}

    def predict(self, history):
        """
        history: list of (x, y)
        """

        # convert to tuple so it can be used as key
        key = tuple(tuple(p) for p in history)

        # return cached if exists
        if key in self.cache:
            return self.cache[key]

        obs = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(self.device)

        # normalize
        origin = obs[:, -1:, :].clone()
        obs = obs - origin

        with torch.no_grad():
            pred = self.model(obs)

        pred = pred + origin
        pred = pred.squeeze(0).cpu().numpy()

        # store in cache
        self.cache[key] = pred

        return pred