#!/usr/bin/python3

import socket
import struct
import time
from datetime import datetime

import store

#listner config
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
BUFFER_SIZE = 4096

#helper function to parse incoming UDP packets
def parse_packet(data):
    #remove incomplete sample at end
    remainder = len(data) % 2
    if remainder:
        data = data[:-remainder]

    #if no data left, return empty list
    count = len(data) // 2
    if count == 0:
        return []

    #return unpacked data as list of shorts
    return struct.unpack(f'<{count}h', data)

def listener(until=None):
    #open UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2*1024*1024)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"Listening on {UDP_IP}:{UDP_PORT}")
    first_packet = True

    try:
        #look forever or until specified time
        while until is None or datetime.now() < until:
            try:
                #receive data
                data, addr = sock.recvfrom(BUFFER_SIZE)

                if first_packet:
                    print(f"Connected to {addr}")
                    first_packet = False

                timestamp = time.time()
                #parse incoming data
                values = parse_packet(data)
                #skip if no data
                if not values:
                    continue

                #add the samples as a list of (timestamp, value) tuples
                samples = [(timestamp, v) for v in values]

                #store the samples thread-safely
                with store.write_lock:
                    store.write_queue.extend(samples)

            #if no data received, just continue
            except socket.timeout:
                continue

    #on exit, close the socket
    finally:
        sock.close()