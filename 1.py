import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Modulation Model
def modulation_model(t, HEF_baseline, A_mod, m):
    HEF_total = HEF_baseline + m * A_mod
    return HEF_total

# Initialize the figure and axes
fig, ax = plt.subplots()
t = np.linspace(0, 10, 1000)
HEF_baseline = np.sin(t)
A_mod = np.cos(t)
m = 0.5

# Plot the initial data
line1, = ax.plot(t, HEF_baseline, label="HEF Baseline")
line2, = ax.plot(t, A_mod, label="Aura Modulating Signal")
line3, = ax.plot(t, modulation_model(t, HEF_baseline, A_mod, m), label="Total HEF")
ax.legend()

# Update function for animation
def update(frame):
    new_HEF_baseline = np.sin(t + frame * 0.1)  # Example update, you can modify this according to your needs
    line1.set_ydata(new_HEF_baseline)

    new_A_mod = np.cos(t + frame * 0.1)
    line2.set_ydata(new_A_mod)

    new_HEF_total = modulation_model(t, new_HEF_baseline, new_A_mod, m)
    line3.set_ydata(new_HEF_total)

    return line1, line2, line3

# Set up the animation
ani = FuncAnimation(fig, update, frames=range(100), interval=100, blit=True)

plt.title("Real-time Modulation Model")
plt.show()

