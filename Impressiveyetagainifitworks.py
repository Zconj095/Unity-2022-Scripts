import datetime
import numpy as np
from ClaudeProject1 import Mitochondria, Synapse

def calculate_biorhythms(date, birth_date, birth_time):
    # Functions from Version1.py
    season = get_season(date)  
    moon_phase = get_moon_phase(date)
    sun_cycle_phase = get_sun_cycle_approx(date)
    circadian_tendency = get_circadian_tendency(birth_time)

    print(f"Season: {season}")
    print(f"Moon Phase: {moon_phase}")
    print(f"Sun Cycle: {sun_cycle_phase}") 
    print(f"Circadian: {circadian_tendency}")

    # Create ClaudeProject1 objects
    mitochondria = Mitochondria()
    synapse = Synapse()

    print(f"ATP Production: {mitochondria.produce_ATP()}")
    print(f"Synapse Receptors: {synapse.num_receptors}")

def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"

def get_moon_phase(date):
    # Simplified moon phase calculation
    days_since_new_moon = (date - datetime.datetime(date.year, date.month, 1)).days
    moon_phase_index = (days_since_new_moon % 29.53) / 29.53
    phases = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous", "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"]
    return phases[int(moon_phase_index * len(phases))]

def get_sun_cycle_approx(date):
    # Simplified sun cycle calculation
    day_of_year = date.timetuple().tm_yday
    if day_of_year <= 182:
        return "Ascending"
    else:
        return "Descending"

def get_circadian_tendency(birth_time):
    # Simplified circadian rhythm calculation based on birth time
    if birth_time.hour < 12:
        return "Morning person"
    else:
        return "Evening person"

today = datetime.datetime(2024, 1, 24)
birth_info = datetime.datetime(1995, 3, 6 , 7, 0, 0)

calculate_biorhythms(today, birth_info, birth_info.time())

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import ephem
import datetime

# Statistical Analysis
def statistical_analysis(data):
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    return mean, median, std_dev

# Time Series Analysis
def time_series_analysis(dates, values):
    plt.plot(dates, values)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Time Series Analysis')
    plt.show()

# Data Visualization
def data_visualization(x, y):
    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Data Visualization')
    plt.show()

def moon_phase_on_date(date):
    observer = ephem.Observer()
    observer.date = date
    prev_new_moon = ephem.previous_new_moon(date)
    next_new_moon = ephem.next_new_moon(date)
    return ephem.localtime(prev_new_moon), ephem.localtime(next_new_moon)

# Sample Data
sample_data = np.random.normal(0, 1, 100)
mean, median, std_dev = statistical_analysis(sample_data)
print(f"Mean: {mean}, Median: {median}, Standard Deviation: {std_dev}")

# Time Series Sample
dates = [datetime.datetime.now() - datetime.timedelta(days=i) for i in range(100)]
values = np.random.normal(0, 1, 100)
time_series_analysis(dates, values)

# Data Visualization Sample
x = np.random.rand(100)
y = np.random.rand(100)
data_visualization(x, y)

# Astronomical Calculation Sample
date = datetime.datetime.now()
prev_new_moon, next_new_moon = moon_phase_on_date(date)
print(f"Previous New Moon: {prev_new_moon}, Next New Moon: {next_new_moon}")
