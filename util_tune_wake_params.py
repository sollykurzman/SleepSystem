#!/usr/bin/python3

import pandas as pd
import numpy as np
import glob
import os
import sys
import itertools

# fixed parameters derived from previous onset tuning
FIXED_ONSET_TH    = 0.81
FIXED_ONSET_DUR   = 50
FIXED_ONSET_ALPHA = 0.105

# initial search ranges for wake parameter tuning
INITIAL_THRESH_RANGE = (0.10, 0.60, 0.05)   
INITIAL_DUR_RANGE    = (30, 600, 30)        
INITIAL_ALPHA_RANGE  = (0.10, 1.0, 0.05)   

def load_ground_truth(night_id):
    path = f"Data/{night_id}/true_sleep_data-{night_id}.csv"
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        # filter for relevant sleep states
        mask = ~df['sleep_state'].isin(['Awake', 'Disturbance', 'notInBed'])
        sleep_df = df[mask]
        
        if sleep_df.empty: return None
        
        # identifying the last recorded moment of sleep
        row = sleep_df.iloc[-1]
        return pd.to_datetime(f"{row['date']} {row['time']}")
    except:
        return None

def load_simulation_data(night_id):
    path = f"Data/{night_id}/classification-{night_id}.csv"
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # extract probability based on available columns
        if 'asleep__Asleep' in df.columns:
            df['prob'] = df['asleep__Asleep']
        elif 'state__Core Sleep' in df.columns:
            cols = ['state__Core Sleep', 'state__Deep Sleep', 'state__REM Sleep']
            existing = [c for c in cols if c in df.columns]
            df['prob'] = df[existing].sum(axis=1)
        else:
            return None
            
        df['prob'] = df['prob'].fillna(0.0)
        return df[['datetime', 'prob']]
    except:
        return None

def simulate_wake_detection(df, wake_alpha, wake_thresh, wake_dur):
    # initialize state variables
    ema = 0.0
    timer_start = None
    is_asleep = False
    
    probs = df['prob'].values
    times = df['datetime'].values
    
    # pre-calculate time deltas for efficiency
    onset_delta = np.timedelta64(FIXED_ONSET_DUR, 's')
    wake_delta  = np.timedelta64(int(wake_dur), 's')
    
    # iterate through time series
    for i in range(len(probs)):
        p = probs[i]
        ts = times[i]
        
        # apply asymmetric smoothing based on current state
        current_alpha = wake_alpha if is_asleep else FIXED_ONSET_ALPHA
        
        # update exponential moving average
        ema = p if i == 0 else (p * current_alpha) + (ema * (1.0 - current_alpha))
        
        if not is_asleep:
            # check for sleep onset using fixed parameters
            if ema >= FIXED_ONSET_TH:
                if timer_start is None: timer_start = ts
                if (ts - timer_start) >= onset_delta:
                    is_asleep = True
                    timer_start = None
            else:
                timer_start = None
        else:
            # check for wake event using tunable parameters
            if ema <= wake_thresh:
                if timer_start is None: timer_start = ts
                if (ts - timer_start) >= wake_delta:
                    return ts 
            else:
                timer_start = None
            
    return None

def calculate_error_minutes(detected_ts, actual_ts):
    # apply heavy penalty if no wake detected
    if detected_ts is None:
        return 180.0 
    # return absolute error in minutes
    return abs((detected_ts - actual_ts).total_seconds()) / 60.0

def generate_grid(center, step, count, min_val, max_val, is_int=False):
    # calculate grid boundaries
    start = center - (step * count)
    end = center + (step * count)
    
    # create range and clip to limits
    grid = np.arange(start, end + (step/1000.0), step)
    grid = np.clip(grid, min_val, max_val)
    
    # format values based on type
    if is_int:
        grid = np.unique(np.round(grid).astype(int))
    else:
        grid = np.unique(np.round(grid, 3))
        
    return grid[grid > 0]

def find_best_params(dataset, thresholds, durations, alphas):
    best_mae = float('inf')
    best_p = (0, 0, 0, float('inf'))
    
    # generate all parameter combinations
    combinations = list(itertools.product(thresholds, durations, alphas))
    total = len(combinations)
    
    # iterate through combinations
    for idx, (t, d, a) in enumerate(combinations):
        errors = []
        # test current combination against all nights
        for entry in dataset:
            detected = simulate_wake_detection(entry['data'], a, t, d)
            errors.append(calculate_error_minutes(detected, entry['truth']))
            
        avg_err = np.mean(errors)
        
        # update best result if error is lower
        if avg_err < best_mae:
            best_mae = avg_err
            best_p = (t, d, a, avg_err)
            
        # log progress
        if idx % 50 == 0 or idx == total - 1:
            sys.stdout.write(f"\r  Scanned {idx+1}/{total}. Best Err: {best_mae:.2f}m (Th={best_p[0]:.2f}, D={best_p[1]}, A={best_p[2]:.2f})")
            sys.stdout.flush()
            
    print("") 
    return best_p

def run_iterative_search():
    print(f"--- Tuning Wake Logic (Fixed Onset: T={FIXED_ONSET_TH}, D={FIXED_ONSET_DUR}s) ---")
    
    # load and validate dataset
    dataset = []
    files = glob.glob("Data/*/classification-*.csv")
    for f in files:
        nid = f.split(os.sep)[1]
        truth = load_ground_truth(nid)
        data = load_simulation_data(nid)
        if truth is not None and data is not None and len(data) > 100:
            dataset.append({'id': nid, 'truth': truth, 'data': data})
    
    print(f"Loaded {len(dataset)} nights.")
    if not dataset: sys.exit(1)

    # perform initial broad survey
    print("\n[Pass 1] Broad Survey...")
    thresholds = np.arange(*INITIAL_THRESH_RANGE)
    durations  = np.arange(*INITIAL_DUR_RANGE)
    alphas     = np.arange(*INITIAL_ALPHA_RANGE)
    
    best_params = find_best_params(dataset, thresholds, durations, alphas)
    print(f"  > Best Pass 1: Th={best_params[0]:.2f}, D={best_params[1]}, A={best_params[2]:.2f} (Err: {best_params[3]:.2f}m)")

    # refine search around best previous results
    print("\n[Pass 2] Refining...")
    thresholds = generate_grid(best_params[0], 0.05, 3, 0.05, 0.95, is_int=False)
    durations  = generate_grid(best_params[1], 15,  4, 15,   900,  is_int=True)
    alphas     = generate_grid(best_params[2], 0.05, 3, 0.05, 0.8,  is_int=False)
    
    best_params = find_best_params(dataset, thresholds, durations, alphas)
    print(f"  > Best Pass 2: Th={best_params[0]:.2f}, D={best_params[1]}, A={best_params[2]:.2f} (Err: {best_params[3]:.2f}m)")

    # perform final micro-optimization
    print("\n[Pass 3] Micro-Optimization...")
    thresholds = generate_grid(best_params[0], 0.01, 4, 0.05, 0.95, is_int=False)
    durations  = generate_grid(best_params[1], 5,   4, 5,   900,  is_int=True)
    alphas     = generate_grid(best_params[2], 0.01, 4, 0.05, 0.8,  is_int=False)
    
    best_params = find_best_params(dataset, thresholds, durations, alphas)
    
    # output final converged parameters
    print("\n" + "="*40)
    print(" CONVERGED WAKE PARAMETERS ")
    print("="*40)
    print(f"Final Mean Error:   {best_params[3]:.3f} minutes")
    print(f"WAKE_THRESH:        {best_params[0]:.3f}")
    print(f"WAKE_DURATION_SEC:  {best_params[1]}")
    print(f"WAKE_ALPHA:         {best_params[2]:.3f}")
    print("="*40)
    print("Ensure you update action.py with DYNAMIC ALPHA logic!")

if __name__ == "__main__":
    run_iterative_search()