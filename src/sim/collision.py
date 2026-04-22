import math

def is_collision(robot_pos, human_positions, threshold=20):
    for hx, hy in human_positions:
        dist = math.sqrt((robot_pos[0] - hx)**2 + (robot_pos[1] - hy)**2)
        if dist < threshold:
            return True
    return False