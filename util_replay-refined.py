#!/usr/bin/python3

import pandas as pd
import os
from tqdm import tqdm

import store
import classifier as classifier
import util_process as data_processor

# ---------------- CONFIG ----------------

PACKETS = 12
BATCH_SIZE = 50 * PACKETS   # same as live system

# ---------------- REPLAY ----------------

def replay_night(raw_csv_path, night_id):
    print(f"\nReplaying night {night_id}")
    print(f"Loading raw data from {raw_csv_path}")

    df = pd.read_csv(raw_csv_path)

    # --- Convert stored CSV → raw samples ---
    if "timestamp_unix" not in df.columns:
        df["timestamp_unix"] = (
            pd.to_datetime(df["datetime"]).astype("int64") / 1e9
        )
        df["adc_raw"] = df["voltage"] / data_processor.VOLTAGE_SCALE

    raw_samples = list(zip(df["timestamp_unix"], df["adc_raw"]))
    total_samples = len(raw_samples)

    print(f"Loaded {total_samples} samples")

    # Reset rolling buffer
    store.live_buffer.buffer.clear()

    accumulator = []

    # tqdm tracks sample progress (not batches)
    for i in tqdm(
        range(0, total_samples, BATCH_SIZE),
        desc="Replaying samples",
        unit="samples",
        unit_scale=BATCH_SIZE
    ):
        accumulator.extend(raw_samples[i:i + BATCH_SIZE])

        if len(accumulator) >= BATCH_SIZE:
            processed_df = data_processor.process_batch(accumulator)
            accumulator.clear()

            if processed_df is not None and not processed_df.empty:
                store.live_buffer.add_batch(processed_df)

                # 🔥 classify immediately (no sleep)
                result = classifier.classify_once(night_id, timestamp=processed_df["datetime"].iloc[-1], use_refined=True, write=False)
                if result:
                    classifier.write_classification_row(night_id, result, csv_path=f"Data/{night_id}/classification_refined-{night_id}.csv")

    print("Replay finished")

# ---------------- ENTRY ----------------

directory = 'Data/'

if __name__ == "__main__":
    nights = []
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder)):
            if os.path.exists(directory + folder + f"/raw_data-{folder}.csv") and not os.path.exists(directory + folder + f"/classification_refined-{folder}.csv"):
                nights.append(folder)
    print(nights)
    for night in nights:
        raw_data_path = f"Data/{night}/raw_data-{night}.csv"
        replay_night(raw_data_path, night)