import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

class RangeSlider:
    def __init__(self, x, window_size=20):
        self.x = x
        self.window_size = window_size

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.2)

        # Setup slider axes
        self.ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(self.ax_slider, 'Range idx', 0, x.shape[1]-1, valinit=0, valfmt='%d')
        self.slider.on_changed(self.update)

        self.update(0)
        plt.show()

    def update(self, val):
        col_idx = int(self.slider.val)
        start = max(0, col_idx - self.window_size // 2)
        end = min(self.x.shape[1], start + self.window_size)

        self.ax.clear()
        # Plot the window of columns
        for col in range(start, end):
            self.ax.plot(self.x.real[:, col], self.x.imag[:, col], '-o', alpha=0.5)

        self.ax.set_title(f"Range idx: {start} to {end} (Centered: {col_idx})")
        self.ax.grid(True)
        self.fig.canvas.draw_idle()

# Usage example (you can call this from your main script/notebook)
# from radar_viewer import RangeSlider
# slider = RangeSlider(your_radar_data)
