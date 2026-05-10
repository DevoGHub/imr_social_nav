from src.planner.dwa import DWAPlanner


class LSTMDWAPlanner(DWAPlanner):
    def __init__(self, **kwargs):
        # Force prediction on; all other params inherit DWA defaults
        kwargs.setdefault("use_prediction", True)
        # Slightly higher clearance weight to leverage richer cost signal
        kwargs.setdefault("w_clearance", 1.8)
        super().__init__(**kwargs)
