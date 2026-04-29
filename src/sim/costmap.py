import math

RADIUS = 80  # tune this

def compute_cost(point, human_positions, human_histories=None, predictor=None):
    cost = 0.0

    px, py = point  # unpack once (faster)

    use_prediction = predictor is not None and human_histories is not None

    for i, (hx, hy) in enumerate(human_positions):

        # Radius filtering
        dx = px - hx
        dy = py - hy
        dist_to_human = math.sqrt(dx*dx + dy*dy)

        if dist_to_human > RADIUS:
            continue

        # CASE 1: No prediction (greedy / normal A*)
        if not use_prediction:
            dist = max(dist_to_human, 1.0)
            cost += math.exp(-dist / 20)

        # CASE 2: Prediction enabled (LSTM)
        else:
            # use precomputed if available
            if hasattr(predictor, "precomputed") and predictor.precomputed is not None:
                pred_traj = predictor.precomputed[i]
            else:
                pred_traj = predictor.predict(human_histories[i])

            for t, (fx, fy) in enumerate(pred_traj[:6]):
                dx = px - fx
                dy = py - fy
                dist = math.sqrt(dx*dx + dy*dy)
                dist = max(dist, 1.0)

                if dist < 10:
                    cost += 1000

                weight = 0.9 ** t
                cost += weight * 50 * math.exp(-dist / 10)

    return cost