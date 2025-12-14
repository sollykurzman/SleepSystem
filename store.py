import os
import time
import threading
from datetime import datetime
from collections import deque

import util_process as data_processor

PACKETS = 12
BATCH_SIZE = 50 * PACKETS

write_queue = deque(maxlen=200_000)
write_lock = threading.Lock()

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

live_buffer = RollingBuffer(window_seconds=30, sample_rate=100)

def store_worker(night_id, until=None):
    print("Store worker started")
    accumulator = []

    os.makedirs(f"Data/{night_id}", exist_ok=True)
    csv_path = f"Data/{night_id}/raw_data-{night_id}.csv"

    while until is None or datetime.now() < until:
        with write_lock:
            if write_queue:
                accumulator.extend(write_queue)
                write_queue.clear()

        if len(accumulator) >= BATCH_SIZE:
            batch = accumulator[:]
            accumulator.clear()

            try:
                df = data_processor.process_batch(batch)
                if df is None or df.empty:
                    continue

                live_buffer.add_batch(df)

                write_header = not os.path.exists(csv_path)
                df.to_csv(csv_path, mode='a', header=write_header, index=False)

            except Exception as e:
                print(f"Processing error: {e}")
        else:
            time.sleep(0.01)

def start_workers(night_id, until=None):
    t = threading.Thread(
        target=store_worker,
        args=(night_id, until),
        daemon=True
    )
    t.start()

# #!/usr/bin/python3

# import struct
# import time
# import os
# from datetime import datetime
# import threading
# from collections import deque

# import util_process as data_processor

# DATE = "Awake"
# UDP_IP = "0.0.0.0"
# UDP_PORT_CAPACITANCE = 5005
# UDP_PORT_STATE = 5006
# BUFFER_SIZE = 4096
# PACKETS = 12
# ML_INTERVAL = 2.0
# STORE_CLASSIFICATION = False

# write_queue = deque(maxlen=200000)
# write_lock = threading.Lock()

# class RollingBuffer:
#     def __init__(self, window_seconds, sample_rate):
#         self.max_len = int(window_seconds * sample_rate)
#         self.buffer = deque(maxlen=self.max_len)
#         self.lock = threading.Lock()
        
#     def add_batch(self, df):
#         if df is None or df.empty:
#             return
        
#         new_data = list(zip(df['datetime'], df['voltage']))
        
#         with self.lock:
#             self.buffer.extend(new_data)
            
#     def get_snapshot(self):
#         with self.lock:
#             data_snapshot = list(self.buffer)

#         if len(data_snapshot) < self.max_len:
#             return None, None
        
#         times, voltages = zip(*data_snapshot)
#         return list(times), list(voltages)

# #Create Buffer Instance
# live_buffer = RollingBuffer(window_seconds=30, sample_rate=100)

# def store_data(date, until=None):
#     print("Processing worker started")
#     accumulator = []

#     while until is None or datetime.now() < until:
#         new_data_batch = []
#         with write_lock:
#             if write_queue:
#                 new_data_batch = list(write_queue)
#                 write_queue.clear()

#         if new_data_batch:
#             accumulator.extend(new_data_batch)
        
#         if len(accumulator) >= (50 * PACKETS):
#             batch = list(accumulator)
#             accumulator.clear()

#             try:
#                 processed_df = data_processor.process_batch(batch)
        
#                 if processed_df is not None and not processed_df.empty:
#                     live_buffer.add_batch(processed_df)

#                     file_exists = os.path.isfile(f"Data/{date}/raw_data-{date}.csv")

#                     processed_df.to_csv(
#                         f"Data/{date}/raw_data-{date}.csv", 
#                         mode='a', 
#                         header=not file_exists, 
#                         index=False
#                     )

#             except Exception as e:
#                 print(f"Error in processing: {e}")

#         elif not new_data_batch:
#             time.sleep(0.01)

# def parse_packet(data):
#     remainder = len(data) % 2
#     if remainder != 0:
#         data = data[:-remainder] 
        
#     count = len(data) // 2
#     if count == 0:
#         return []
        
#     return struct.unpack(f'<{count}h', data)

# def start_workers(date, until=None):
#     if not os.path.exists(f"Data/{date}"):
#         os.makedirs(f"Data/{date}")

#     args = (date, until)
    
#     data_thread = threading.Thread(target=store_data, args=args, daemon=True)
#     data_thread.start()
