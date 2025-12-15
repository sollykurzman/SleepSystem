#!/usr/bin/python3

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm

import classifier
import action
import db
import store
import main

# list of night ids to simulate
SIM_NIGHT_ID = ["131125", "141125", "151125", "171125", "181125", "201125", "211125", "221125", "241125", "261125", "281125", "291125", "301125", "011225", "051225"]
BASE_DB_NAME = "sleep.db"

class MockBuffer:
    def __init__(self):
        self.times = []
        self.volts = []

    def set_window(self, times, volts):
        self.times = times
        self.volts = volts

    def get_snapshot(self):
        return self.times, self.volts

def run_simulation_worker(night_id):
    # record start time for performance tracking
    start_timer = time.time()
    
    # construct path and validate data existence
    csv_path = f"Data/{night_id}/raw_data-{night_id}.csv"
    if not os.path.exists(csv_path):
        return f"Error: Could not find {csv_path}", []

    # load and sort csv data
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # create unique database file for this process to avoid locking
    worker_db_name = f"sleep_{night_id}.db"
    if os.path.exists(worker_db_name):
        os.remove(worker_db_name)
        
    # patch database path in db module
    db.DB_PATH = Path(worker_db_name)
    db.init_db()

    # initialize night context based on data start time
    cutoff_time = df['datetime'].iloc[0].time()
    context = main.build_night_context(cutoff_time)
    context.night_id = night_id

    # reset classifier state and inject mock buffer
    classifier.reset_state()
    mock_buffer = MockBuffer()
    store.live_buffer = mock_buffer

    # define simulation parameters
    WINDOW_SEC = 30
    STEP_SEC = 2
    
    # convert dataframe columns to numpy arrays for faster indexing
    all_times = df['datetime'].values
    all_volts = df['voltage'].values
    
    # define time boundaries
    start_time = df['datetime'].iloc[0]
    end_time = df['datetime'].iloc[-1]
    total_seconds = (end_time - start_time).total_seconds()
    
    current_head = start_time + timedelta(seconds=WINDOW_SEC)
    
    # initialize loop variables
    idx_start = 0
    idx_end = 0
    n_samples = len(all_times)
    
    window_td = np.timedelta64(WINDOW_SEC, 's')
    step_count = 0
    
    # set logging frequency
    LOG_INTERVAL = 900 

    # loop through dataset sequentially
    while current_head < end_time:
        # print progress update periodically
        if step_count % LOG_INTERVAL == 0 and step_count > 0:
            progress = ((current_head - start_time).total_seconds() / total_seconds) * 100
            print(f"  [Night {night_id}] {current_head.strftime('%H:%M')} ({progress:.0f}%)")

        head_np = np.datetime64(current_head)
        tail_np = head_np - window_td
        
        # advance end index to current head time
        while idx_end < n_samples and all_times[idx_end] < head_np:
            idx_end += 1
            
        # advance start index to maintain window size
        while idx_start < idx_end and all_times[idx_start] < tail_np:
            idx_start += 1
            
        # process window if sufficient data exists
        if idx_end - idx_start > 100: 
            mock_buffer.set_window(
                all_times[idx_start:idx_end],
                all_volts[idx_start:idx_end]
            )

            # run classification logic
            result = classifier.classify_once(
                night_id=night_id, 
                write=True,
                timestamp=current_head,
                use_refined=True
            )

            # execute action logic if result valid
            if result:
                action.handle_classification(
                    context=context,
                    ts=result['datetime'],
                    label=result['classification'],
                    probabilities=result['probabilities']
                )

        # advance simulation time
        current_head += timedelta(seconds=STEP_SEC)
        step_count += 1

    # query recorded events from temporary database
    conn = db.get_conn()
    events = conn.execute("SELECT ts, type, payload FROM sleep_events WHERE night_id=?", (night_id,)).fetchall()
    
    # structure event data
    event_list = []
    for e in events:
        event_list.append({
            'ts': e['ts'],
            'type': e['type'],
            'payload': e['payload']
        })
    conn.close()
    
    # delete temporary database file
    # if os.path.exists(worker_db_name):
    #     os.remove(worker_db_name)

    # calculate statistics and return summary
    elapsed = time.time() - start_timer
    summary = f"Night {night_id}: {len(event_list)} events, {step_count} steps ({elapsed:.1f}s)"
    
    return summary, event_list

def merge_worker_dbs(worker_db_files, final_db_path="sleep.db"):
    # point db module at final database
    db.DB_PATH = Path(final_db_path)
    db.init_db()
    conn = db.get_conn()

    for worker_db in worker_db_files:
        print(f"Merging {worker_db} into {final_db_path}")

        conn.execute(f"ATTACH DATABASE '{worker_db}' AS worker")

        # adjust table list if you have more than sleep_events
        conn.execute("""
            INSERT INTO sleep_events
            SELECT * FROM worker.sleep_events
        """)

        conn.execute("DETACH DATABASE worker")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    # determine worker count based on cpu availability
    max_workers = min(len(SIM_NIGHT_ID), os.cpu_count())
    print(f"Starting simulation on {len(SIM_NIGHT_ID)} nights using {max_workers} cores...")
    
    total_start = time.time()
    
    # execute simulations in parallel
    worker_dbs = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_simulation_worker, nid): nid for nid in SIM_NIGHT_ID}

        for future in tqdm(as_completed(futures), total=len(SIM_NIGHT_ID), desc="Simulating Nights"):
            night_id = futures[future]
            try:
                summary, _ = future.result()
                tqdm.write(summary)

                worker_dbs.append(f"sleep_{night_id}.db")

            except Exception as e:
                tqdm.write(f"Night {night_id} generated an exception: {e}")

    print("Merging worker databases into sleep.db...")
    merge_worker_dbs(worker_dbs, BASE_DB_NAME)

    print(f"Total simulation time: {time.time() - total_start:.2f}s")