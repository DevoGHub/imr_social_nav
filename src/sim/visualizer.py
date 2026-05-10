import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()

    def render(self, state, title=None):
        self.ax.clear()

        humans = state["humans"]
        robot = state["robot"]
        goal = state["goal"]

        # plot humans
        if humans:
            xs, ys = zip(*humans)
            self.ax.scatter(xs, ys, c="blue", label="Humans")

        # plot robot
        self.ax.scatter(robot[0], robot[1], c="red", label="Robot")

        # Plot Goal
        self.ax.scatter(goal[0], goal[1], c="green", label="Goal")

        self.ax.legend()
        MODE_LABELS = {
            "greedy":     "Greedy Simulation",
            "astar":      "Astar Simulation",
            "lstm_astar": "LSTM A* Simulation",
            "dwa":        "DWA Simulation",
            "lstm_dwa":   "LSTM DWA Simulation",
        }
        display_title = MODE_LABELS.get(title, title) if title else "Simulation"
        self.ax.set_title(display_title)

        plt.draw()
        plt.pause(0.01)