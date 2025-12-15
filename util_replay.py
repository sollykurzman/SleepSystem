#!/usr/bin/python3

import pandas as pd
import os
import time
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

import store
import classifier as classifier
import util_process as data_processor

# configuration settings
PACKETS = 12
BATCH_SIZE = 50 * PACKETS
REFINED_MODEL = True       

@contextmanager
def suppress_stdout():
    # redirect stdout to null to hide spammy library outputs
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def save_results_batch(night_id, results, output_path):
    if not results:
        return

    # define columns for output structure
    ALL_COLUMNS = {
        "in_bed": ["inBed", "notInBed"],
        "asleep": ["Asleep", "Awake"],
        "state": ["Core Sleep", "REM Sleep", "Deep Sleep"]
    }

    processed_rows = []

    # format each result row for csv
    for res in results:
        row = {
            "datetime": res["datetime"],
            "classification": res["classification"]
        }
        
        # extract and flatten nested probabilities
        for stage, classes in ALL_COLUMNS.items():
            probs = res["probabilities"].get(stage, {})
            for cls in classes:
                row[f"{stage}__{cls}"] = probs.get(cls, None)
        
        processed_rows.append(row)

    # write formatted data to csv
    df_out = pd.DataFrame(processed_rows)
    df_out.to_csv(output_path, index=False)


def process_night_wrapper(args):
    # unpack arguments for concurrent execution
    raw_csv_path, night_id, use_refined = args
    return replay_night(raw_csv_path, night_id, use_refined)

def replay_night(raw_csv_path, night_id, use_refined):
    # silence initialization logs
    with suppress_stdout():
        classifier.reset_state() 
        store.live_buffer.buffer.clear()

    # read raw data file
    try:
        df = pd.read_csv(raw_csv_path)
    except Exception as e:
        return f"Error reading {night_id}: {e}"

    # normalize columns if needed
    if "timestamp_unix" not in df.columns:
        df["timestamp_unix"] = pd.to_datetime(df["datetime"]).astype("int64") / 1e9
        df["adc_raw"] = df["voltage"] / data_processor.VOLTAGE_SCALE

    # convert dataframe to list of samples
    raw_samples = list(zip(df["timestamp_unix"], df["adc_raw"]))
    total_samples = len(raw_samples)
    
    accumulator = []
    night_results = []
    
    # initialize progress timers
    last_print_time = time.time()
    print_interval = 5.0 

    # iterate through all samples in batches
    for i in range(0, total_samples, BATCH_SIZE):
        
        # check if it is time to print progress
        current_time = time.time()
        if current_time - last_print_time > print_interval:
            progress_pct = (i / total_samples) * 100
            print(f"[{night_id}] {progress_pct:.0f}% processed ({len(night_results)} classifications)")
            last_print_time = current_time

        # add batch to accumulator
        batch = raw_samples[i:i + BATCH_SIZE]
        accumulator.extend(batch)

        # process accumulator when full
        if len(accumulator) >= BATCH_SIZE:
            processed_df = data_processor.process_batch(accumulator)
            accumulator.clear()

            # update live buffer and classify
            if processed_df is not None and not processed_df.empty:
                store.live_buffer.add_batch(processed_df)

                timestamp = processed_df["datetime"].iloc[-1]
                
                # classify without printing internal logs
                with suppress_stdout():
                    result = classifier.classify_once(
                        night_id, 
                        timestamp=timestamp, 
                        use_refined=use_refined, 
                        write=False 
                    )

                if result:
                    night_results.append(result)

    # determine output filename based on model used
    if use_refined:
        out_name = f"classification-{night_id}.csv"
    else:
        out_name = f"classification_prerefinement-{night_id}.csv"
        
    out_path = os.path.join("Data", night_id, out_name)
    
    # save all classification results
    save_results_batch(night_id, night_results, out_path)

    return f"Finished Night {night_id} ({len(night_results)} classifications)"

directory = 'Data/'

if __name__ == "__main__":
    tasks = []
    
    # verify data directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found.")
        exit()

    print("Scanning Data/ directory...")
    
    # identify folders requiring processing
    for folder in sorted(os.listdir(directory)):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            raw_path = os.path.join(folder_path, f"raw_data-{folder}.csv")
            
            # determine expected output file path
            if REFINED_MODEL:
                out_file = os.path.join(folder_path, f"classification-{folder}.csv")
            else:
                out_file = os.path.join(folder_path, f"classification_prerefinement-{folder}.csv")

            # add task if raw data exists and output is missing
            if os.path.exists(raw_path) and not os.path.exists(out_file):
                tasks.append((raw_path, folder, REFINED_MODEL))

    print(f"Found {len(tasks)} nights to process.")

    max_workers = os.cpu_count()
    print(f"Spinning up pool with {max_workers} workers...")
    print("------------------------------------------------")

    start_total = time.time()

    # execute tasks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_night_wrapper, tasks), 
            total=len(tasks), 
            desc="Nights Completed",
            unit="night"
        ))

    print("\n------------------------------------------------")
    for res in results:
        print(res)

    print(f"\nAll done in {time.time() - start_total:.2f} seconds.")