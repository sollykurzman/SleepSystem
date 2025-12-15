#!/usr/bin/python3

import sqlite3
import pandas as pd
import json

DB_NAME = "simulation.db"

def inspect():
    # log inspection start
    print(f"--- Inspecting {DB_NAME} ---")
    
    # establish database connection
    conn = sqlite3.connect(DB_NAME)

    # log section header
    print("\n=== SLEEP EVENTS ===")
    try:
        # retrieve all sleep events
        df_events = pd.read_sql_query("SELECT * FROM sleep_events", conn)
        
        # display events if available
        if not df_events.empty:
            print(df_events[['ts', 'type', 'payload']].to_string(index=False))
        else:
            print("No events found.")
    except Exception as e:
        # handle query errors
        print(f"Error reading events: {e}")

    # log section header
    print("\n=== ALARMS ===")
    try:
        # retrieve all alarm records
        df_alarms = pd.read_sql_query("SELECT * FROM alarms", conn)
        
        # display alarms if available
        if not df_alarms.empty:
            print(df_alarms[['ts', 'target_ts', 'status', 'reason']].to_string(index=False))
        else:
            print("No alarms found.")
    except Exception as e:
        # handle query errors
        print(f"Error reading alarms: {e}")

    # log section header
    print("\n=== CLASSIFICATIONS (First 5 & Last 5) ===")
    try:
        # retrieve classification history
        df_class = pd.read_sql_query("SELECT * FROM classifications", conn)
        
        if not df_class.empty:
            # display total count
            print(f"Total classifications stored: {len(df_class)}")
            
            # display first few records
            print("\n--- First 5 ---")
            print(df_class[['ts', 'label', 'probabilities']].head(5).to_string(index=False))
            
            # display most recent records
            print("\n--- Last 5 ---")
            print(df_class[['ts', 'label', 'probabilities']].tail(5).to_string(index=False))
        else:
            print("No classifications found.")
    except Exception as e:
        # handle query errors
        print(f"Error reading classifications: {e}")

    # close database connection
    conn.close()

if __name__ == "__main__":
    inspect()