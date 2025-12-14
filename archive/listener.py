import socket
import struct
import time
from datetime import datetime

import store

UDP_IP = "0.0.0.0"
UDP_PORT = 5005
BUFFER_SIZE = 4096

def parse_packet(data):
    remainder = len(data) % 2
    if remainder:
        data = data[:-remainder]

    count = len(data) // 2
    if count == 0:
        return []

    return struct.unpack(f'<{count}h', data)

def listener(until=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2*1024*1024)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"Listening on {UDP_IP}:{UDP_PORT}")
    first_packet = True

    try:
        while until is None or datetime.now() < until:
            try:
                data, addr = sock.recvfrom(BUFFER_SIZE)

                if first_packet:
                    print(f"Connected to {addr}")
                    first_packet = False

                timestamp = time.time()
                values = parse_packet(data)
                if not values:
                    continue

                samples = [(timestamp, v) for v in values]

                with store.write_lock:
                    store.write_queue.extend(samples)

            except socket.timeout:
                continue

    finally:
        sock.close()


# import socket
# import struct
# import time
# import pandas as pd
# from datetime import datetime

# # import util_process as data_processor
# import store


# UDP_IP = "0.0.0.0"
# UDP_PORT = 5005
# PACKETS = 12
# BUFFER_SIZE = 4096

# def parse_packet(data):
#     remainder = len(data) % 2
#     if remainder != 0:
#         data = data[:-remainder] 
        
#     count = len(data) // 2
#     if count == 0:
#         return []
        
#     return struct.unpack(f'<{count}h', data)

# def listener(until=None, night_id=None):

#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.settimeout(1.0)
#     sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2*1024*1024)
#     sock.bind((UDP_IP, UDP_PORT))
    
#     print(f"Listening on {UDP_IP}:{UDP_PORT}...")
#     first_packet = True

#     local_accumulator = []
#     BATCH_THRESHOLD = 50 * PACKETS

#     try:
#         while until is None or datetime.now() < until:
#             try:
#                 data, addr = sock.recvfrom(BUFFER_SIZE)

#                 if first_packet:
#                     print(f"Connected to {addr[0]}:{addr[1]}")
#                     first_packet = False
#                     # if night_id:
#                     #     save_event_to_json(
#                     #         "receiver_first_packet",
#                     #         datetime.now(),
#                     #         file_path=f"Data/{night_id}/sleep_events-{night_id}.json"
#                     #     )
                
#                 timestamp = time.time()
#                 adc_values = parse_packet(data)
                
#                 if not adc_values:
#                     continue

#                 new_entries = [(timestamp, val) for val in adc_values]
                
#                 local_accumulator.extend(new_entries)

#                 with store.write_lock:
#                     store.write_queue.extend(new_entries)

#                 # if len(local_accumulator) >= BATCH_THRESHOLD:
                    
#                     # try:
#                     #     processed_df = data_processor.process_batch(local_accumulator)
 
#                         # if processed_df is not None and not processed_df.empty:
#                         #     classifier.live_buffer.add_batch(processed_df)
                            
#                     # except Exception as e:
#                     #     print(f"Error processing batch: {e}")
#                         # if night_id:
#                         #     log_error_to_json(
#                         #         f"Error processing batch: {e}",
#                         #         file_path=f"Data/{night_id}/sleep_events-{night_id}.json"
#                         #     )

#                     local_accumulator.clear()

#             except socket.timeout:
#                 if not first_packet: print("No data...")
#                 continue

#     except KeyboardInterrupt:
#         print("Stopping...")
#         sock.close()