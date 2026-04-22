import math

def compute_cost(point, human_positions):
    cost = 0.0

    for hx, hy in human_positions:
        dist = math.sqrt((point[0] - hx)**2 + (point[1] - hy)**2)

        if dist < 1:
            dist = 1  # avoid division by zero

        # exponential decay for social discomfort
        cost += math.exp(-dist / 20)

    return cost