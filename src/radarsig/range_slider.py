import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from typing import Callable, Optional


class RangeSlider:
    def __init__(
        self,
        data: np.ndarray,
        plot_callback: Callable[[plt.Axes, np.ndarray, Optional[np.ndarray]], None],
        axis: int = 0,
        x: Optional[np.ndarray] = None,
        window_size: int = 20,
    ):
        self.data = data
        self.plot_callback = plot_callback
        self.axis = axis
        self.x = x
        self.window_size = window_size

        # Ensure we handle the axis correctly for the slider range
        self.dim_size = data.shape[axis]

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.2)

        # Setup slider axes
        self.ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        # The slider moves along the specified axis
        self.slider = Slider(
            self.ax_slider,
            f"Axis {axis} idx",
            0,
            self.dim_size - 1,
            valinit=0,
            valfmt="%d",
            valstep=1,
        )
        self.slider.on_changed(self.update)

        # Setup keyboard support
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.update(0)
        plt.show()

    def on_key(self, event):
        if event.key == "left":
            self.step_prev(event)
        elif event.key == "right":
            self.step_next(event)

    def step_prev(self, event):
        new_val = max(0, int(self.slider.val) - 1)
        self.slider.set_val(new_val)

    def step_next(self, event):
        new_val = min(self.dim_size - 1, int(self.slider.val) + 1)
        self.slider.set_val(new_val)

    def update(self, val):
        idx = int(self.slider.val)
        start = max(0, idx - self.window_size // 2)
        end = min(self.dim_size, start + self.window_size)

        # Create a slice of the data based on axis and window
        if self.axis == 0:
            data_slice = self.data[start:end, :]
        else:
            data_slice = self.data[:, start:end]

        self.ax.clear()

        # Call the callback to plot
        self.plot_callback(self.ax, data_slice, self.x)

        self.ax.set_title(f"Axis {self.axis} idx: {start} to {end} (Centered: {idx})")
        self.ax.grid(True)
        self.fig.canvas.draw_idle()
