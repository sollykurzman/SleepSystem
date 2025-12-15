#!/usr/bin/python3

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

TARGET_SAMPLE_RATE = 100
ANTIALIASING_CUTOFF = 45.0
FILTER_ORDER = 5
VOLTAGE_SCALE = 0.125 / 1000.0


def redistribute_timestamps_linear(df):
    # identify unique timestamp markers
    unique_timestamps = df['timestamp_unix'].unique()
    
    # return original if insufficient data for interpolation
    if len(unique_timestamps) < 2:
        return df

    # calculate average duration between packets
    avg_gap = np.mean(np.diff(unique_timestamps))

    new_rows = []
    
    # group data by original timestamp
    grouped = df.groupby('timestamp_unix')
    
    for i, ts in enumerate(unique_timestamps):
        group = grouped.get_group(ts)
        n_samples = len(group)
        
        # determine duration to next packet
        if i < len(unique_timestamps) - 1:
            next_ts = unique_timestamps[i+1]
            time_gap = next_ts - ts
        else:
            # use average gap for the final packet
            time_gap = avg_gap

        # generate linear time offsets for samples within the packet
        if n_samples > 1:
            offsets = np.linspace(0, time_gap, n_samples, endpoint=False)
        else:
            offsets = np.array([0.0])

        # apply offsets to spread timestamps evenly
        current_data = group.copy()
        current_data['timestamp_unix'] = ts + offsets
        new_rows.append(current_data)

    # combine redistributed segments
    if new_rows:
        return pd.concat(new_rows).reset_index(drop=True)
    return df

def process_batch(raw_batch):
    # check for empty input
    if not raw_batch:
        return None

    # convert buffer to dataframe
    df = pd.DataFrame(raw_batch, columns=['timestamp_unix', 'adc_raw'])
    
    # scale raw adc values to voltage
    df['voltage'] = df['adc_raw'] * VOLTAGE_SCALE
    
    # smooth out timestamp distribution
    df = redistribute_timestamps_linear(df)
    
    # calculate total duration
    time_span = df['timestamp_unix'].max() - df['timestamp_unix'].min()
    if time_span == 0: return None
    
    # estimate current sampling frequency
    fs_est = len(df) / time_span

    # define nyquist requirement for filter
    min_required_fs = ANTIALIASING_CUTOFF * 2

    # apply low-pass filter if sampling rate is sufficient
    if fs_est > min_required_fs and len(df) > FILTER_ORDER * 4:
        b, a = butter(FILTER_ORDER, ANTIALIASING_CUTOFF / (fs_est / 2), btype='low')
        try:
            df['voltage'] = filtfilt(b, a, df['voltage'].values)
        except Exception as e:
            print(f"Filter Error (skipping filter): {e}")
    else:
        print(f"Warning: Sampling rate {fs_est:.2f}Hz too low for 45Hz filter. Skipping.")
    
    # convert unix timestamps to datetime objects
    df['datetime'] = pd.to_datetime(df['timestamp_unix'], unit='s')
    df = df.set_index('datetime')
    
    # calculate resample interval string
    resample_interval = f"{1000 // TARGET_SAMPLE_RATE}ms"
    
    # downsample to target rate using mean
    df_final = df['voltage'].resample(resample_interval).mean().dropna().reset_index()
    
    return df_final