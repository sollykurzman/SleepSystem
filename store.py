#!/usr/bin/python3

import os
import time
import threading
from datetime import datetime
from collections import deque

import util_process as data_processor

PACKETS = 12
BATCH_SIZE = 50 * PACKETS

#define data queue and lock
write_queue = deque(maxlen=200_000)
write_lock = threading.Lock()

#buffer to hold recent live data
class RollingBuffer:
    def __init__(self, window_seconds, sample_rate):
        self.max_len = int(window_seconds * sample_rate)
        self.buffer = deque(maxlen=self.max_len)
        self.lock = threading.Lock()

    def add_batch(self, df):
        if df is None or df.empty:
            return
        with self.lock:
            self.buffer.extend(zip(df['datetime'], df['voltage']))

    def get_snapshot(self):
        with self.lock:
            if len(self.buffer) < self.max_len:
                return None, None
            t, v = zip(*self.buffer)
            return list(t), list(v)

#create buffer instance
live_buffer = RollingBuffer(window_seconds=30, sample_rate=100)

#worker function to store incoming data
def store_worker(night_id, until=None):
    print("Store worker started")
    accumulator = []

    #ensure data directory exists
    os.makedirs(f"Data/{night_id}", exist_ok=True)
    csv_path = f"Data/{night_id}/raw_data-{night_id}.csv"

    #loop until time limit or forever
    while until is None or datetime.now() < until:
        #wait until lock is free
        with write_lock:
            if write_queue:
                #transfer data from write queue to accumulator
                accumulator.extend(write_queue)
                #clear the write queue
                write_queue.clear()

        #process data if enough in accumulator
        if len(accumulator) >= BATCH_SIZE:
            #create a copy of accumulator and clear it
            batch = accumulator[:]
            accumulator.clear()

            try:
                #process the data until data frame for classifier
                df = data_processor.process_batch(batch)
                if df is None or df.empty:
                    continue
                
                #add to live buffer
                live_buffer.add_batch(df)

                #if no file, write header and data, otherwise append data
                write_header = not os.path.exists(csv_path)
                df.to_csv(csv_path, mode='a', header=write_header, index=False)

            except Exception as e:
                print(f"Processing error: {e}")
        
        #if accumulator is not full, pause and then try again
        else:
            time.sleep(0.01)
