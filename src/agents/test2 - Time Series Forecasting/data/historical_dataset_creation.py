import csv
import numpy as np
from datetime import datetime, timedelta

# Base distribution parameters
mu_base = 5  # base mean occupancy
sigma = 2  # base standard deviation


def hourly_occupancy(hour):
    # General occupancy trend by hour
    if 9 <= hour <= 13 or 17 <= hour <= 20:
        return mu_base + 6
    elif 14 <= hour <= 16:
        return mu_base + 2
    else:
        return mu_base


with open(r'historical_occupancy.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ds', 'y'])
    start = datetime(2025, 7, 1, 0, 0)

    for i in range(62 * 24):  # July + August by hour
        ts = start + timedelta(hours=i)

        # Base trend by hour
        mu = hourly_occupancy(ts.hour)

        # --- Extra noise to break the obvious pattern ---
        daily_shift = np.random.normal(0, 3)  # day-to-day variation
        weekly_shift = np.random.uniform(-2, 2)  # slower weekly variation
        burst_noise = np.random.poisson(1) * 0.5  # occasional random spikes
        jitter = np.random.normal(0, 0.8)  # small extra jitter

        # Adjusted mean
        mu_adjusted = mu + daily_shift + weekly_shift + burst_noise

        # Dynamic sigma (more variability on weekends)
        sigma_dynamic = sigma * (1.5 if ts.weekday() >= 5 else 1)
        occ = int(max(0, np.random.normal(mu_adjusted, sigma_dynamic) + jitter))

        writer.writerow([ts.strftime('%Y-%m-%d %H:%M:%S'), occ])