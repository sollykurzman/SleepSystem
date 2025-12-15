import pandas as pd
import numpy as np
import glob
import os
import sys
import itertools

# configuration for parameter grid search
INITIAL_THRESH_RANGE = (0.10, 0.95, 0.05) 
INITIAL_DUR_RANGE    = (20, 900, 20)    
INITIAL_ALPHA_RANGE  = (0.01, 0.20, 0.05) 

def load_ground_truth(night_id):
    path = f"Data/{night_id}/true_sleep_data-{night_id}.csv"
    if not os.path.exists(path): return None
    try:
        # load csv and filter for sleep states
        df = pd.read_csv(path)
        mask = ~df['sleep_state'].isin(['Awake', 'Disturbance', 'notInBed'])
        sleep_df = df[mask]
        
        # return the first timestamp of confirmed sleep
        if sleep_df.empty: return None
        row = sleep_df.iloc[0]
        return pd.to_datetime(f"{row['date']} {row['time']}")
    except:
        return None

def load_simulation_data(night_id):
    path = f"Data/{night_id}/classification-{night_id}.csv"
    if not os.path.exists(path): return None
    try:
        # load classification results
        df = pd.read_csv(path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # calculate sleep probability based on available columns
        if 'asleep__Asleep' in df.columns:
            df['prob'] = df['asleep__Asleep']
        elif 'state__Core Sleep' in df.columns:
            cols = ['state__Core Sleep', 'state__Deep Sleep', 'state__REM Sleep']
            existing = [c for c in cols if c in df.columns]
            df['prob'] = df[existing].sum(axis=1)
        else:
            return None
            
        # fill missing probabilities with zero
        df['prob'] = df['prob'].fillna(0.0)
        return df[['datetime', 'prob']]
    except:
        return None

def simulate_night(df, alpha, thresh, duration):
    # initialize state variables
    ema = 0.0
    timer_start = None
    probs = df['prob'].values
    times = df['datetime'].values
    
    # pre-calculate duration delta for comparison
    duration_delta = np.timedelta64(int(duration), 's')
    
    # simulate real-time processing
    for i in range(len(probs)):
        p = probs[i]
        
        # update exponential moving average
        ema = p if i == 0 else (p * alpha) + (ema * (1.0 - alpha))
            
        # check if probability exceeds threshold
        if ema >= thresh:
            if timer_start is None:
                timer_start = times[i]
            
            # check if duration condition is met
            if (times[i] - timer_start) >= duration_delta:
                return times[i] 
        else:
            # reset timer if threshold condition fails
            timer_start = None
            
    return None

def calculate_error_minutes(detected_ts, actual_ts):
    # apply penalty if no onset detected
    if detected_ts is None:
        return 300.0 
    
    # return absolute error in minutes
    return abs((detected_ts - actual_ts).total_seconds()) / 60.0

def generate_grid(center, step, count, min_val, max_val, is_int=False):
    # define search boundaries around center point
    start = center - (step * count)
    end = center + (step * count)
    
    # create grid array
    grid = np.arange(start, end + (step/1000.0), step) 
    
    # clip values to valid range
    grid = np.clip(grid, min_val, max_val)
    
    # apply rounding if needed
    if is_int:
        grid = np.unique(np.round(grid).astype(int))
    else:
        grid = np.unique(np.round(grid, 3))
        
    return grid[grid > 0] 

def run_iterative_search():
    print("--- Loading Data ---")
    dataset = []
    
    # load data from all night directories
    files = glob.glob("Data/*/classification-*.csv")
    for f in files:
        nid = f.split(os.sep)[1]
        truth = load_ground_truth(nid)
        data = load_simulation_data(nid)
        
        # add valid nights to dataset
        if truth is not None and data is not None and len(data) > 100:
            dataset.append({'id': nid, 'truth': truth, 'data': data})
    
    print(f"Loaded {len(dataset)} nights.")
    if not dataset: sys.exit(1)

    # perform initial broad parameter sweep
    print("\n[Pass 1] Broad Survey...")
    thresholds = np.arange(*INITIAL_THRESH_RANGE)
    durations  = np.arange(*INITIAL_DUR_RANGE)
    alphas     = np.arange(*INITIAL_ALPHA_RANGE)
    
    best_params = find_best_params(dataset, thresholds, durations, alphas)
    print(f"  > Best Pass 1: T={best_params[0]}, D={best_params[1]}, A={best_params[2]} (Err: {best_params[3]:.2f}m)")

    # perform refined parameter sweep around best results
    print("\n[Pass 2] Refining...")
    thresholds = generate_grid(best_params[0], 0.05, 3, 0.5, 0.95, is_int=False)
    durations  = generate_grid(best_params[1], 30,  4, 60,  1200, is_int=True)
    alphas     = generate_grid(best_params[2], 0.02, 3, 0.01, 0.5,  is_int=False)
    
    best_params = find_best_params(dataset, thresholds, durations, alphas)
    print(f"  > Best Pass 2: T={best_params[0]}, D={best_params[1]}, A={best_params[2]} (Err: {best_params[3]:.2f}m)")

    # perform final micro-optimization
    print("\n[Pass 3] Micro-Optimization...")
    thresholds = generate_grid(best_params[0], 0.01, 3, 0.5, 0.99, is_int=False)
    durations  = generate_grid(best_params[1], 10,  3, 30,  1200, is_int=True)
    alphas     = generate_grid(best_params[2], 0.005, 3, 0.005, 0.5, is_int=False)
    
    best_params = find_best_params(dataset, thresholds, durations, alphas)
    
    # display final converged parameters
    print("\n" + "="*40)
    print(" CONVERGED RESULTS ")
    print("="*40)
    print(f"Final Mean Error:   {best_params[3]:.3f} minutes")
    print(f"ONSET_THRESH:       {best_params[0]}")
    print(f"ONSET_DURATION_SEC: {best_params[1]}")
    print(f"EMA_ALPHA:          {best_params[2]}")
    print("="*40)

def find_best_params(dataset, thresholds, durations, alphas):
    best_mae = float('inf')
    best_p = (0, 0, 0, float('inf'))
    
    # generate all parameter combinations
    combinations = list(itertools.product(thresholds, durations, alphas))
    total = len(combinations)
    
    # iterate through combinations
    for idx, (t, d, a) in enumerate(combinations):
        errors = []
        # test combination against full dataset
        for entry in dataset:
            detected = simulate_night(entry['data'], a, t, d)
            errors.append(calculate_error_minutes(detected, entry['truth']))
            
        avg_err = np.mean(errors)
        
        # update best result if error is lower
        if avg_err < best_mae:
            best_mae = avg_err
            best_p = (t, d, a, avg_err)
            
        # log progress periodically
        if idx % 10 == 0 or idx == total - 1:
            sys.stdout.write(f"\r  Scanned {idx+1}/{total} configs. Current Best: {best_mae:.2f}m")
            sys.stdout.flush()
            
    print("") 
    return best_p