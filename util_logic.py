#!/usr/bin/python3

from datetime import datetime, timedelta, time as dt_time
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import json

import db
from util_calendar import get_calendar_data 
from scheduler import scheduler

HOURS_GOAL = 8.0

def calculate_sleep_debt():
    # fetch sleep history for the last week
    history = db.get_sleep_history(days=7)
    if not history:
        return 0.0

    total_sleep_seconds = 0
    last_onset = None

    # calculate total sleep duration from onset and wake events
    for event in history:
        if event['type'] == 'sleep_onset':
            last_onset = event['ts']
        elif event['type'] in ['wake_up', 'alarm_target'] and last_onset:
            # calculate duration of the sleep segment
            duration = (event['ts'] - last_onset).total_seconds()
            
            # sanity check to ignore negative durations or excessively long periods
            if 0 < duration < 57600: 
                total_sleep_seconds += duration
                last_onset = None 
                
    # compare actual sleep against the 8-hour goal
    days_count = 7
    total_needed = days_count * HOURS_GOAL * 3600
    
    debt_seconds = total_needed - total_sleep_seconds
    return debt_seconds / 3600.0

def determine_initial_alarm(night_context):
    # identify calendar constraints for the wake-up date
    search_date = night_context.until.date()
    cal_events = get_calendar_data(search_date)
    
    # set default wake-up time if calendar is empty
    target_time = datetime.combine(search_date, dt_time(10, 0))
    
    if cal_events:
        # filter valid events and sort by time
        valid_events = [e for e in cal_events if 'ignorethis' not in e.get('notes', '')]
        valid_events.sort(key=lambda x: x['time'])
        
        if valid_events:
            first_event_dt = datetime.combine(search_date, valid_events[0]['time'])
            # buffer wake-up time one hour before first event
            target_time = first_event_dt - timedelta(hours=1)

    # set the determined alarm time in the scheduler
    scheduler.set_alarm(target_time)
    print(f"Logic: Initial alarm set for {target_time}")

def get_calendar_limit(context):
    # fetch calendar events for the target date
    events = get_calendar_data(context.until.date())

    events = sorted(events, key=lambda x: x['time'])

    # filter out ignored events
    events = [
        entry for entry in events
        if 'notes' in entry and 'ignorethis' not in entry['notes']
    ]

    # default to 10am if no events exist
    if not events:
        return datetime.strptime("10:00:00", "%H:%M:%S").time()

    event_time = datetime.combine(datetime.today(), events[0]['time'])

    # set limit one hour prior to first event
    first_event = event_time - timedelta(hours=1)

    return first_event.time()

def run_smart_adjustments(night_context):
    # get currently scheduled alarm time
    current_alarm = scheduler.get_schedule()
    
    # abort if alarm is missing or imminent
    if not current_alarm:
        return
    if (current_alarm - datetime.now()).total_seconds() < 2700:
        return

    # fetch recent classification history for cycle detection
    rows = db.get_recent_classifications(night_context.night_id, hours=4)

    # ensure sufficient data points exist
    if len(rows) < 100: 
        return

    # parse probability data into a dataframe
    data = []
    for r in rows:
        try:
            probs = json.loads(r['probabilities'])
            
            # combine core and deep sleep probabilities
            core_p = probs.get('state', {}).get('Core Sleep', 0.0)
            deep_p = probs.get('state', {}).get('Deep Sleep', 0.0)
            
            data.append({
                'ts': pd.to_datetime(r['ts']),
                'core_prob': core_p + deep_p
            })
        except Exception:
            continue

    if not data: return
    df = pd.DataFrame(data).set_index('ts')

    # resample data to minute intervals and fill gaps
    resampled = df['core_prob'].resample('1min').mean().fillna(0)
    
    # smooth signal using a rolling average
    smooth_signal = resampled.rolling(window=20, center=True).mean().fillna(0)

    # identify peaks representing core sleep cycles
    peaks, _ = find_peaks(
        smooth_signal.values, 
        distance=60,       
        height=0.3,        
        prominence=0.1     
    )

    if len(peaks) < 1:
        return

    # estimate timing of the next sleep cycle peak
    peak_times = smooth_signal.index[peaks]
    last_peak_time = peak_times[-1]
    
    if len(peaks) >= 2:
        intervals = pd.Series(peak_times).diff().dropna()
        avg_cycle_mins = intervals.dt.total_seconds().mean() / 60
        # clamp cycle length to physiological norms
        if avg_cycle_mins < 70 or avg_cycle_mins > 120:
            avg_cycle_mins = 90
    else:
        avg_cycle_mins = 90

    predicted_peak = last_peak_time + timedelta(minutes=avg_cycle_mins)

    # check for collision between alarm and deep sleep peak
    collision_window_minutes = 30
    time_diff = abs((current_alarm - predicted_peak).total_seconds() / 60)
    
    if time_diff <= collision_window_minutes:
        # shift alarm earlier to avoid waking during deep sleep
        new_target_time = predicted_peak - timedelta(minutes=20)
        
        # apply adjustment if valid
        if new_target_time > datetime.now():
            print(f"SMART: Moving alarm from {current_alarm.strftime('%H:%M')} to {new_target_time.strftime('%H:%M')} (Avoids peak at {predicted_peak.strftime('%H:%M')})")
            
            scheduler.set_alarm(new_target_time)
            
            # log adjustment event to database
            db.insert_event(
                night_context.night_id, 
                "smart_alarm_adjustment", 
                payload={
                    "old_time": str(current_alarm),
                    "new_time": str(new_target_time),
                    "reason": f"Projected Peak collision at {predicted_peak}"
                }
            )