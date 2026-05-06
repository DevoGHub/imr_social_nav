"""
LSTM-Guided Dynamic Window Approach (LSTM DWA) Planner
--------------------------------------------------------
Extends DWAPlanner by enabling LSTM-based pedestrian trajectory
predictions in the costmap scoring. The only difference from plain
DWA is that use_prediction=True is set by default, which causes
_get_cost() to pass human_histories and predictor through to
compute_cost(), inflating the dynamic window cost around predicted
future pedestrian positions rather than only current ones.
"""

from src.planner.dwa import DWAPlanner


class LSTMDWAPlanner(DWAPlanner):
    def __init__(self, **kwargs):
        # Force prediction on; all other params inherit DWA defaults
        kwargs.setdefault("use_prediction", True)
        # Slightly higher clearance weight to leverage richer cost signal
        kwargs.setdefault("w_clearance", 1.8)
        super().__init__(**kwargs)
