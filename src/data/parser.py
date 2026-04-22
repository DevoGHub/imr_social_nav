from collections import defaultdict

def parse_annotation_line(line):
    parts = line.strip().split()

    if len(parts) < 10:
        return None

    track_id = int(parts[0])
    xmin, ymin = float(parts[1]), float(parts[2])
    xmax, ymax = float(parts[3]), float(parts[4])
    frame = int(parts[5])
    lost = int(parts[6])
    occluded = int(parts[7])
    generated = int(parts[8])
    label = parts[9].strip('"')

    # Keep only pedestrians
    if label != "Pedestrian":
        return None

    # Remove invalid frames
    if lost == 1:
        return None

    # Convert bbox to center point
    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0

    return track_id, frame, x, y


def load_annotations(file_path):
    data = []

    with open(file_path, "r") as f:
        for line in f:
            parsed = parse_annotation_line(line)
            if parsed is not None:
                data.append(parsed)

    return data


def build_trajectories(file_path):
    annotations = load_annotations(file_path)

    trajectories = defaultdict(list)

    for track_id, frame, x, y in annotations:
        trajectories[track_id].append((frame, x, y))

    # sort each trajectory by frame
    for track_id in trajectories:
        trajectories[track_id].sort(key=lambda t: t[0])

    return trajectories


def clean_trajectories(trajectories, min_len=20):
    cleaned = {}

    for track_id, traj in trajectories.items():
        if len(traj) >= min_len:
            cleaned[track_id] = traj

    return cleaned