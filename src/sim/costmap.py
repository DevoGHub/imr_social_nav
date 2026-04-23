import math

def compute_cost(point, human_positions, human_histories=None, predictor=None):
    cost = 0.0

    for i, (hx, hy) in enumerate(human_positions):

        # CASE 1: No prediction
        if predictor is None or human_histories is None:
            dist = math.sqrt((point[0] - hx)**2 + (point[1] - hy)**2)

            if dist < 1:
                dist = 1

            cost += math.exp(-dist / 20)

        # CASE 2: Prediction enabled
        else:
            history = human_histories[i]
            pred_traj = predictor.predict(history)  # shape: (T, 2)

            for t, (px, py) in enumerate(pred_traj):
                dist = math.sqrt((point[0] - px)**2 + (point[1] - py)**2)

                if dist < 1:
                    dist = 1

                # decay future importance
                weight = 0.9 ** t

                cost += weight * math.exp(-dist / 20)

    return cost