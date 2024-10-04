import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal

# Parameters
num_channels = 5  
window_size = 100
samp_rate = 250

# Initialize mock EEG data
eeg_data = np.random.normal(size=(num_channels, window_size))  

# Initialize figure 
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.set_ylim(0, 20)
ax1.set_title('Alpha Power')
ax1.set_xlabel('Time')
ax1.set_ylabel('μV^2')

ax2.set_ylim(0, 30)
ax2.set_title('Peak Frequency')
ax2.set_xlabel('Time')  
ax2.set_ylabel('Hz')

alpha_line, = ax1.plot([], [])
freq_line, = ax2.plot([], [])

# Update functions  
def update_alpha(data):
    power = np.sum(signal.welch(data)[1][8:13]) 
    return power

def update_freq(data):
    peak = signal.find_peaks(np.fft.fft(data))[0][0] * samp_rate / window_size
    return peak  

# Animation function
def animate(i):

    # Shift and update EEG buffer 
    eeg_data[:,:-1] = eeg_data[:,1:]  
    eeg_data[:,-1] = np.random.normal(size=num_channels)

    # Extract features
    alpha = update_alpha(eeg_data[0])
    freq = update_freq(eeg_data[0])

    # Update plots
    alpha_line.set_data(range(i), alpha)
    freq_line.set_data(range(i), freq)
    
    return alpha_line, freq_line

ani = FuncAnimation(fig, animate, interval=50)
plt.show()
#### Run auric field models #### 



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
num_samples = 100
feedback_coefficient = 0.8
haptic_response_time = 0.1

# Generate random XYZ data
xyz_data = np.random.rand(num_samples, 3)
haptic_feedback_xyz = np.zeros_like(xyz_data)

# Simulate haptic feedback response
for i in range(1, num_samples):
    haptic_feedback_xyz[i] = feedback_coefficient * haptic_feedback_xyz[i-1] + (1 - feedback_coefficient) * xyz_data[i-1]

# Convert XYZ to polar coordinates
r = np.linalg.norm(xyz_data, axis=1)
theta = np.arctan2(xyz_data[:, 1], xyz_data[:, 0])
phi = np.arccos(xyz_data[:, 2] / r)

# Convert haptic feedback XYZ to polar coordinates
r_haptic = np.linalg.norm(haptic_feedback_xyz, axis=1)
theta_haptic = np.arctan2(haptic_feedback_xyz[:, 1], haptic_feedback_xyz[:, 0])
phi_haptic = np.arccos(haptic_feedback_xyz[:, 2] / r_haptic)

# 3D plot (XYZ vs. Haptic Feedback XYZ)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz_data[:, 0], xyz_data[:, 1], xyz_data[:, 2], label='XYZ Data')
ax.scatter(haptic_feedback_xyz[:, 0], haptic_feedback_xyz[:, 1], haptic_feedback_xyz[:, 2], label='Haptic Feedback XYZ', marker='x')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
plt.title('Triangularity between XYZ and Haptic Feedback XYZ')
plt.show()

# 2D polar plot (Polar vs. Haptic Feedback Polar)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r, label='Polar Data')
ax.plot(theta_haptic, r_haptic, label='Haptic Feedback Polar', linestyle='dashed')
ax.set_rlabel_position(0)
ax.legend()
plt.title('Triangularity between Polar and Haptic Feedback Polar')
plt.show()

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Parameters
num_samples = 100

# Generate two vectors representing time frames
vector_time_frame1 = np.random.rand(num_samples)
vector_time_frame2 = 0.8 * vector_time_frame1 + 0.2 * np.random.rand(num_samples)

# Calculate the correlation coefficient
correlation_coefficient, _ = pearsonr(vector_time_frame1, vector_time_frame2)

# Scatter plot with correlation coefficient
plt.scatter(vector_time_frame1, vector_time_frame2, label=f'Correlation: {correlation_coefficient:.2f}')
plt.xlabel('Vector Time Frame 1')
plt.ylabel('Vector Time Frame 2')
plt.legend()
plt.title('Linear Response Feedback from Haptic Feedback')
plt.show()

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Parameters
text_data = "responsive time is a measure of the time it takes to respond to a signal"
response_time_data = np.random.rand(100)

# Calculate signature response time using CuPy
signature_response_time = cp.mean(cp.asarray(response_time_data))

# Word Cloud plot
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')

# Response time plot
plt.figure()
plt.plot(response_time_data, label='Response Time')
plt.xlabel('Sample')
plt.ylabel('Response Time')
plt.legend()
plt.title('Signature Response Time Analysis')

# Show the plots
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_dimensions = 3
num_samples = 1000
frequency_pattern = [1.5, 2.5, 3.0]

# Time values
t = np.linspace(0, 10, num_samples)

# Simulate hyperflux data with frequency patterns
hyperflux_data = np.zeros((num_samples, num_dimensions))
for dim in range(num_dimensions):
    hyperflux_data[:, dim] = np.sin(2 * np.pi * frequency_pattern[dim] * t)

# Hyperflux data plot
fig, axs = plt.subplots(num_dimensions, 1, figsize=(10, 6), sharex=True)
fig.suptitle('Hyperflux Data with Frequency Patterns')

for dim in range(num_dimensions):
    axs[dim].plot(t, hyperflux_data[:, dim], label=f'Dimension {dim+1}')
    axs[dim].legend()
    axs[dim].set_ylabel('Amplitude')

axs[num_dimensions-1].set_xlabel('Time')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Modulation model function
def modulation_model(t, HEF_baseline, A_mod, m):
    HEF_total = HEF_baseline + m * A_mod
    return HEF_total

# Create a figure and axis
fig, ax = plt.subplots()
t = np.linspace(0, 10, 1000)
HEF_baseline = np.sin(t)
A_mod = np.cos(t)
m = 0.5

# Initial plot setup
line1, = ax.plot(t, HEF_baseline, label="HEF Baseline")
line2, = ax.plot(t, A_mod, label="Aura Modulating Signal")
line3, = ax.plot(t, modulation_model(t, HEF_baseline, A_mod, m), label="Total HEF")
ax.legend()

# Animation update function
def update_modulation(frame):
    new_HEF_baseline = np.sin(t + frame * 0.1)
    line1.set_ydata(new_HEF_baseline)

    new_A_mod = np.cos(t + frame * 0.1)
    line2.set_ydata(new_A_mod)

    new_HEF_total = modulation_model(t, new_HEF_baseline, new_A_mod, m)
    line3.set_ydata(new_HEF_total)

    return line1, line2, line3

# Create animation
ani_modulation = FuncAnimation(fig, update_modulation, frames=range(100), interval=100, blit=True)

plt.title("Real-time Modulation Model")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Coupled oscillators function
def coupled_oscillators(HEF_a, A_a):
    k1, k2, k3, k4 = 0.1, 0.2, 0.3, 0.4

    dHEF_dt = k1 * HEF_a - k2 * A_a
    dA_dt = -k3 * HEF_a + k4 * A_a

    return dHEF_dt, dA_dt

# Create a figure and axis
fig, ax = plt.subplots()
HEF_a0, A_a0 = 1, 0
t = np.linspace(0, 20, 1000)

HEF_a = np.empty_like(t)
A_a = np.empty_like(t)

HEF_a[0] = HEF_a0
A_a[0] = A_a0

# Initial plot setup
line4, = ax.plot(t, HEF_a, label="HEF Amplitude")
line5, = ax.plot(t, A_a, label="Aura Amplitude")
ax.legend()

# Animation update function
def update_coupling(frame):
    new_dHEF_dt, new_dA_dt = coupled_oscillators(HEF_a[frame-1], A_a[frame-1])
    HEF_a[frame] = HEF_a[frame-1] + new_dHEF_dt
    A_a[frame] = A_a[frame-1] + new_dA_dt

    line4.set_ydata(HEF_a)
    line5.set_ydata(A_a)

    return line4, line5

# Create animation
ani_coupling = FuncAnimation(fig, update_coupling, frames=range(1, len(t)), interval=100, blit=True)

plt.title("Real-time Coupling Model")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Information transfer function
def information_transfer(HEF_a, A_a):
    I = k * HEF_a * A_a
    return I

# Create a figure and axis
fig, ax = plt.subplots()
k = 0.1
t = np.linspace(0, 20, 1000)

HEF_a = 1 + 0.5 * np.sin(t)
A_a = 2 + 0.3 * np.cos(t)

I = information_transfer(HEF_a, A_a)

# Initial plot setup
line6, = ax.plot(t, I, label="Information Transfer Rate")
ax.legend()

# Animation update function
def update_information_transfer(frame):
    new_HEF_a = 1 + 0.5 * np.sin(t + frame * 0.1)
    new_A_a = 2 + 0.3 * np.cos(t + frame * 0.1)

    new_I = information_transfer(new_HEF_a, new_A_a)

    line6.set_ydata(new_I)

    return line6,

# Create animation
ani_information_transfer = FuncAnimation(fig, update_information_transfer, frames=range(100), interval=100, blit=True)

plt.title("Real-time Information Transfer Rate")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Modulation model function
def modulation_model(t, HEF_baseline, A_mod, m):
    HEF_total = HEF_baseline + m * A_mod
    return HEF_total

# Create a figure and axis
fig, ax = plt.subplots()
t = np.linspace(0, 10, 1000)
HEF_baseline = np.sin(t)
A_mod = np.cos(t)
m = 0.5

# Initial plot setup
line1, = ax.plot(t, HEF_baseline, label="HEF Baseline")
line2, = ax.plot(t, A_mod, label="Aura Modulating Signal")
line3, = ax.plot(t, modulation_model(t, HEF_baseline, A_mod, m), label="Total HEF")
ax.legend()

# Animation update function
def update_modulation(frame):
    new_HEF_baseline = np.sin(t + frame * 0.1)
    line1.set_ydata(new_HEF_baseline)

    new_A_mod = np.cos(t + frame * 0.1)
    line2.set_ydata(new_A_mod)

    new_HEF_total = modulation_model(t, new_HEF_baseline, new_A_mod, m)
    line3.set_ydata(new_HEF_total)

    return line1, line2, line3

# Create animation
ani_modulation = FuncAnimation(fig, update_modulation, frames=range(100), interval=100, blit=True)

plt.title("Real-time Modulation Model")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
num_dimensions = 3
num_samples = 1000
frequency_pattern = [1.5, 2.5, 3.0]  # Adjust frequencies for each dimension

# Time values
t = np.linspace(0, 10, num_samples)

# Create a figure and axes
fig, axs = plt.subplots(num_dimensions, 1, figsize=(10, 6), sharex=True)
fig.suptitle('Hyperflux Data with Frequency Patterns')

# Initialize hyperflux data
hyperflux_data = np.zeros((num_samples, num_dimensions))
for dim in range(num_dimensions):
    hyperflux_data[:, dim] = np.sin(2 * np.pi * frequency_pattern[dim] * t)

# Plot setup
for dim in range(num_dimensions):
    axs[dim].plot(t, hyperflux_data[:, dim], label=f'Dimension {dim+1}')
    axs[dim].legend()
    axs[dim].set_ylabel('Amplitude')

axs[num_dimensions-1].set_xlabel('Time')

# Animation update function
def update_hyperflux(frame):
    for dim in range(num_dimensions):
        hyperflux_data[:, dim] = np.sin(2 * np.pi * frequency_pattern[dim] * (t + frame * 0.1))
        axs[dim].clear()
        axs[dim].plot(t, hyperflux_data[:, dim], label=f'Dimension {dim+1}')
        axs[dim].legend()
        axs[dim].set_ylabel('Amplitude')

    axs[num_dimensions-1].set_xlabel('Time')

    return axs

# Create animation
ani_hyperflux = FuncAnimation(fig, update_hyperflux, frames=range(100), interval=100, blit=True)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function for coupled oscillators
def coupled_oscillators(HEF_a, A_a):
    k1, k2, k3, k4 = 0.1, 0.2, 0.3, 0.4

    dHEF_dt = k1 * HEF_a - k2 * A_a
    dA_dt = -k3 * HEF_a + k4 * A_a

    return dHEF_dt, dA_dt

# Create a figure and axis
fig, ax = plt.subplots()
HEF_a0, A_a0 = 1, 0
t = np.linspace(0, 20, 1000)

# Initialize arrays
HEF_a = np.empty_like(t)
A_a = np.empty_like(t)

HEF_a[0] = HEF_a0
A_a[0] = A_a0

# Initial plot setup
line4, = ax.plot(t, HEF_a, label="HEF Amplitude")
line5, = ax.plot(t, A_a, label="Aura Amplitude")
ax.legend()

# Animation update function
def update_coupling(frame):
    new_dHEF_dt, new_dA_dt = coupled_oscillators(HEF_a[frame-1], A_a[frame-1])
    HEF_a[frame] = HEF_a[frame-1] + new_dHEF_dt
    A_a[frame] = A_a[frame-1] + new_dA_dt

    line4.set_ydata(HEF_a)
    line5.set_ydata(A_a)

    return line4, line5

# Create animation
ani_coupling = FuncAnimation(fig, update_coupling, frames=range(1, len(t)), interval=100, blit=True)

plt.title("Real-time Coupling Model")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function for information transfer
def information_transfer(HEF_a, A_a):
    I = k * HEF_a * A_a
    return I

# Create a figure and axis
fig, ax = plt.subplots()
k = 0.1
t = np.linspace(0, 20, 1000)

# Initialize arrays
HEF_a = 1 + 0.5 * np.sin(t)
A_a = 2 + 0.3 * np.cos(t)

I = information_transfer(HEF_a, A_a)

# Initial plot setup
line6, = ax.plot(t, I, label="Information Transfer Rate")
ax.legend()

# Animation update function
def update_information_transfer(frame):
    new_HEF_a = 1 + 0.5 * np.sin(t + frame * 0.1)
    new_A_a = 2 + 0.3 * np.cos(t + frame * 0.1)

    new_I = information_transfer(new_HEF_a, new_A_a)

    line6.set_ydata(new_I)

    return line6,

# Create animation
ani_information_transfer = FuncAnimation(fig, update_information_transfer, frames=range(100), interval=100, blit=True)

plt.title("Real-time Information Transfer Rate")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function for custom haptic feedback model
def modulation_model(t, HEF_baseline, A_mod, m):
    HEF_total = HEF_baseline + m * A_mod
    return HEF_total

# Create a figure and axis
fig, ax = plt.subplots()
t = np.linspace(0, 10, 1000)
HEF_baseline = np.sin(t)
A_mod = np.cos(t)
m = 0.5

# Initial plots setup
line1, = ax.plot(t, HEF_baseline, label="HEF Baseline")
line2, = ax.plot(t, A_mod, label="Aura Modulating Signal")
line3, = ax.plot(t, modulation_model(t, HEF_baseline, A_mod, m), label="Total HEF")
ax.legend()

# Animation update function
def update_modulation(frame):
    new_HEF_baseline = np.sin(t + frame * 0.1)
    line1.set_ydata(new_HEF_baseline)

    new_A_mod = np.cos(t + frame * 0.1)
    line2.set_ydata(new_A_mod)

    new_HEF_total = modulation_model(t, new_HEF_baseline, new_A_mod, m)
    line3.set_ydata(new_HEF_total)

    return line1, line2, line3

# Create animation
ani_modulation = FuncAnimation(fig, update_modulation, frames=range(100), interval=100, blit=True)

plt.title("Real-time Modulation Model")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function for coupled oscillators model
def coupled_oscillators(HEF_a, A_a):
    k1, k2, k3, k4 = 0.1, 0.2, 0.3, 0.4

    dHEF_dt = k1 * HEF_a - k2 * A_a
    dA_dt = -k3 * HEF_a + k4 * A_a

    return dHEF_dt, dA_dt

# Create a figure and axis
fig, ax = plt.subplots()
HEF_a0, A_a0 = 1, 0
t = np.linspace(0, 20, 1000)

HEF_a = np.empty_like(t)
A_a = np.empty_like(t)

HEF_a[0] = HEF_a0
A_a[0] = A_a0

# Initial plots setup
line4, = ax.plot(t, HEF_a, label="HEF Amplitude")
line5, = ax.plot(t, A_a, label="Aura Amplitude")
ax.legend()

# Animation update function
def update_coupling(frame):
    new_dHEF_dt, new_dA_dt = coupled_oscillators(HEF_a[frame-1], A_a[frame-1])
    HEF_a[frame] = HEF_a[frame-1] + new_dHEF_dt
    A_a[frame] = A_a[frame-1] + new_dA_dt

    line4.set_ydata(HEF_a)
    line5.set_ydata(A_a)

    return line4, line5

# Create animation
ani_coupling = FuncAnimation(fig, update_coupling, frames=range(1, len(t)), interval=100, blit=True)

plt.title("Real-time Coupling Model")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function for information transfer rate
def information_transfer(HEF_a, A_a):
    k = 0.1
    I = k * HEF_a * A_a
    return I

# Create a figure and axis
fig, ax = plt.subplots()
k = 0.1
t = np.linspace(0, 20, 1000)

HEF_a = 1 + 0.5 * np.sin(t)
A_a = 2 + 0.3 * np.cos(t)

I = information_transfer(HEF_a, A_a)

# Initial plot setup
line6, = ax.plot(t, I, label="Information Transfer Rate")
ax.legend()

# Animation update function
def update_information_transfer(frame):
    new_HEF_a = 1 + 0.5 * np.sin(t + frame * 0.1)
    new_A_a = 2 + 0.3 * np.cos(t + frame * 0.1)

    new_I = information_transfer(new_HEF_a, new_A_a)

    line6.set_ydata(new_I)

    return line6,

# Create animation
ani_information_transfer = FuncAnimation(fig, update_information_transfer, frames=range(100), interval=100, blit=True)

plt.title("Real-time Information Transfer Rate")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function for coupled oscillators
def coupled_oscillators(HEF_a, A_a):
    k1, k2, k3, k4 = 0.1, 0.2, 0.3, 0.4

    dHEF_dt = k1 * HEF_a - k2 * A_a
    dA_dt = -k3 * HEF_a + k4 * A_a

    return dHEF_dt, dA_dt

# Create a figure and axis
fig, ax = plt.subplots()
HEF_a0, A_a0 = 1, 0
t = np.linspace(0, 20, 1000)

HEF_a = np.empty_like(t)
A_a = np.empty_like(t)

HEF_a[0] = HEF_a0
A_a[0] = A_a0

# Initial plot setup
line4, = ax.plot(t, HEF_a, label="HEF Amplitude")
line5, = ax.plot(t, A_a, label="Aura Amplitude")
ax.legend()

# Animation update function
def update_coupling(frame):
    new_dHEF_dt, new_dA_dt = coupled_oscillators(HEF_a[frame-1], A_a[frame-1])
    HEF_a[frame] = HEF_a[frame-1] + new_dHEF_dt
    A_a[frame] = A_a[frame-1] + new_dA_dt

    line4.set_ydata(HEF_a)
    line5.set_ydata(A_a)

    return line4, line5

# Create animation
ani_coupling = FuncAnimation(fig, update_coupling, frames=range(1, len(t)), interval=100, blit=True)

plt.title("Real-time Coupling Model")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function for information transfer
def information_transfer(HEF_a, A_a):
    I = k * HEF_a * A_a
    return I

# Create a figure and axis
fig, ax = plt.subplots()
k = 0.1
t = np.linspace(0, 20, 1000)

HEF_a = 1 + 0.5 * np.sin(t)
A_a = 2 + 0.3 * np.cos(t)

I = information_transfer(HEF_a, A_a)

# Initial plot setup
line6, = ax.plot(t, I, label="Information Transfer Rate")
ax.legend()

# Animation update function
def update_information_transfer(frame):
    new_HEF_a = 1 + 0.5 * np.sin(t + frame * 0.1)
    new_A_a = 2 + 0.3 * np.cos(t + frame * 0.1)

    new_I = information_transfer(new_HEF_a, new_A_a)

    line6.set_ydata(new_I)

    return line6,

# Create animation
ani_information_transfer = FuncAnimation(fig, update_information_transfer, frames=range(100), interval=100, blit=True)

plt.title("Real-time Information Transfer Rate")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function for information transfer
def information_transfer(HEF_a, A_a):
    I = k * HEF_a * A_a
    return I

# Create a figure and axis
fig, ax = plt.subplots()
k = 0.1
t = np.linspace(0, 20, 1000)

HEF_a = 1 + 0.5 * np.sin(t)
A_a = 2 + 0.3 * np.cos(t)

I = information_transfer(HEF_a, A_a)

# Initial plot setup
line6, = ax.plot(t, I, label="Information Transfer Rate")
ax.legend()

# Animation update function
def update_information_transfer(frame):
    new_HEF_a = 1 + 0.5 * np.sin(t + frame * 0.1)
    new_A_a = 2 + 0.3 * np.cos(t + frame * 0.1)

    new_I = information_transfer(new_HEF_a, new_A_a)

    line6.set_ydata(new_I)

    return line6,

# Create animation
ani_information_transfer = FuncAnimation(fig, update_information_transfer, frames=range(100), interval=100, blit=True)

plt.title("Real-time Information Transfer Rate")
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def coupled_oscillators(HEF_a, A_a):
    k1, k2, k3, k4 = 0.1, 0.2, 0.3, 0.4

    dHEF_dt = k1 * HEF_a - k2 * A_a
    dA_dt = -k3 * HEF_a + k4 * A_a

    return dHEF_dt, dA_dt

fig, ax = plt.subplots()
HEF_a0, A_a0 = 1, 0
t = np.linspace(0, 20, 1000)

HEF_a = np.empty_like(t)
A_a = np.empty_like(t)

HEF_a[0] = HEF_a0
A_a[0] = A_a0

line4, = ax.plot(t, HEF_a, label="HEF Amplitude")
line5, = ax.plot(t, A_a, label="Aura Amplitude")
ax.legend()

def update_coupling(frame):
    new_dHEF_dt, new_dA_dt = coupled_oscillators(HEF_a[frame-1], A_a[frame-1])
    HEF_a[frame] = HEF_a[frame-1] + new_dHEF_dt
    A_a[frame] = A_a[frame-1] + new_dA_dt

    line4.set_ydata(HEF_a)
    line5.set_ydata(A_a)

    return line4, line5

ani_coupling = FuncAnimation(fig, update_coupling, frames=range(1, len(t)), interval=100, blit=True)

plt.title("Real-time Coupling Model")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def coupled_oscillators(HEF_a, A_a, k1, k2, k3, k4):
    dHEF_dt = k1 * HEF_a - k2 * A_a
    dA_dt = -k3 * HEF_a + k4 * A_a
    return dHEF_dt, dA_dt

fig, ax = plt.subplots()
HEF_a0, A_a0 = 1, 0
t = np.linspace(0, 20, 1000)

HEF_a = np.empty_like(t)
A_a = np.empty_like(t)

HEF_a[0] = HEF_a0
A_a[0] = A_a0

k1, k2, k3, k4 = 0.1, 0.2, 0.3, 0.4

line4, = ax.plot(t, HEF_a, label="HEF Amplitude")
line5, = ax.plot(t, A_a, label="Aura Amplitude")
ax.legend()

def update_coupling(frame):
    new_dHEF_dt, new_dA_dt = coupled_oscillators(HEF_a[frame-1], A_a[frame-1], k1, k2, k3, k4)
    HEF_a[frame] = HEF_a[frame-1] + new_dHEF_dt
    A_a[frame] = A_a[frame-1] + new_dA_dt

    line4.set_ydata(HEF_a)
    line5.set_ydata(A_a)

    return line4, line5

ani_coupling = FuncAnimation(fig, update_coupling, frames=range(1, len(t)), interval=100, blit=True)

plt.title("Real-time Auric Data Modeling")
plt.show()

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create mock EEG data stream
n_channels = 5
eeg_data = np.random.randn(n_channels, 10)

# Initialize plot 
fig, ax = plt.subplots()
lines = ax.plot(eeg_data)
ax.set_ylim(-5, 5)
ax.set_title('EEG')
ax.set_xlabel('Time')
ax.set_ylabel('Voltage')

def update(frame):
    # Get additional data points     
    new_data = np.random.randn(n_channels)  
    eeg_data[:,:-1] = eeg_data[:,1:] # shift data  
    eeg_data[:,-1] = new_data  # add new 
    
    # Update plot 
    for lnum, line in enumerate(lines):
        line.set_ydata(eeg_data[lnum])
    
    return lines

# Construct the animation
ani = FuncAnimation(fig, update, interval=50)  
plt.show()

# Run auric field models based on eeg_data...


