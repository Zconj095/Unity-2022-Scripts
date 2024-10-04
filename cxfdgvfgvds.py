import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Auric Sensation as Energy Flow

def energy_flow_model(E_a, P_a, S_a):
    A_s = k1 * E_a + k2 * P_a + k3 * S_a
    return A_s

# Initialize the figure and axes
fig, ax = plt.subplots()
E_a = 2
P_a = 0.5
S_a = 0.8
k1, k2, k3 = 0.3, 0.4, 0.5

A_s_energy_flow = energy_flow_model(E_a, P_a, S_a)

# Plot the initial data
line10, = ax.plot(0, A_s_energy_flow, 'bo-', label="Energy Flow")
ax.legend()

# Update function for animation
def update_energy_flow(frame):
    new_E_a = 2 + 0.1 * np.sin(frame * 0.1)  # Example update, modify as needed
    new_P_a = 0.5 + 0.1 * np.cos(frame * 0.1)
    new_S_a = 0.8 + 0.1 * np.sin(frame * 0.1)

    new_A_s_energy_flow = energy_flow_model(new_E_a, new_P_a, new_S_a)

    line10.set_xdata(np.append(line10.get_xdata(), frame))
    line10.set_ydata(np.append(line10.get_ydata(), new_A_s_energy_flow))

    return line10,

# Set up the animation
ani_energy_flow = FuncAnimation(fig, update_energy_flow, frames=range(100), interval=100, blit=True)

plt.title("Real-time Auric Sensation (Energy Flow)")
plt.xlabel("Frame")
plt.ylabel("Auric Sensation")
plt.show()
