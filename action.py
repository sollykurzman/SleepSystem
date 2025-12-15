#!/usr/bin/python3

import threading
from collections import deque
from datetime import datetime, timedelta
import math
import pandas as pd 

import classifier
import db
import scheduler  
import logic      

# threshold configuration
ONSET_THRESH = 0.81
ONSET_DURATION_SEC = 50
WAKE_THRESH = 0.40
WAKE_DURATION_SEC = 120
EMA_ALPHA = 0.105

# smart alarm check configuration
SMART_ADJUST_WINDOW = 60 
LAST_SMART_CHECK = datetime.now()

def action_loop(context):
    global LAST_SMART_CHECK
    
    # log start event
    db.insert_event(context.night_id, "sleep_engine_started")
    
    # set initial alarm based on calendar constraints
    logic.determine_initial_alarm(context)

    while True:
        # retrieve next classification result
        result = classifier.classification_queue.get()
        
        # process classification data
        handle_classification(
            context=context,
            ts=result["datetime"],
            label=result["classification"],
            probabilities=result["probabilities"]
        )
        
        # periodic check for smart alarm adjustments
        if (datetime.now() - LAST_SMART_CHECK).total_seconds() > SMART_ADJUST_WINDOW:
            # analyze recent sleep cycles for optimization
            logic.run_smart_adjustments(context)
            LAST_SMART_CHECK = datetime.now()

def handle_classification(context, ts, label, probabilities):
    # update history buffer
    context.classification_history.append(label)
    
    # store classification record
    db.insert_classification(context.night_id, ts, label, probabilities)

    # calculate sleep probability
    prob_asleep = extract_sleep_probability(probabilities, label)

    # update exponential moving average
    update_ema_state(context, prob_asleep)

    # determine if state change occurred
    event = evaluate_state_transition(context, ts)

    # handle sleep onset or wake events
    if event == "sleep_onset":
        on_sleep_onset(context, ts)
    elif event == "wake_up":
        on_wake(context, ts)

def extract_sleep_probability(probabilities, label):
    try:
        # return zero if user not in bed
        if label == "notInBed":
            return 0.0

        # check for refined model state
        if 'state' in probabilities and isinstance(probabilities['state'], dict):
             # calculate sleep probability from awake score
             p = 1.0 - probabilities.get('asleep', {}).get('Awake', 0.0)
             return float(p)
             
        # check binary model probability
        p = probabilities.get('asleep', {}).get('Asleep', 0.0)
        
        # handle invalid probability values
        if p is None or math.isnan(p):
            return 0.0
            
        return float(p)

    except (AttributeError, TypeError, ValueError):
        # fallback to label-based probability
        return 1.0 if label in ["Asleep", "Core Sleep", "Deep Sleep", "REM"] else 0.0

def update_ema_state(context, current_prob):
    # replace empty or NaN probabilities with zero
    if current_prob is None or math.isnan(current_prob):
        current_prob = 0.0
    
    # initialise ema if missing
    if not hasattr(context, 'prob_ema') or context.prob_ema is None:
        context.prob_ema = current_prob
        return

    # reset ema if corrupted
    if math.isnan(context.prob_ema):
        context.prob_ema = current_prob

    # apply ema formula
    context.prob_ema = (current_prob * EMA_ALPHA) + (context.prob_ema * (1.0 - EMA_ALPHA))

def evaluate_state_transition(context, ts):
    # initialise transition timestamp
    if not hasattr(context, 'state_change_start_ts'):
        context.state_change_start_ts = None

    # get current smoothed probability
    current_ema = getattr(context, 'prob_ema', 0.0)

    # check for onset or wake criteria
    if not context.currently_asleep:
        return check_onset_criteria(context, current_ema, ts)
    else:
        return check_wake_criteria(context, current_ema, ts)

def check_onset_criteria(context, ema, ts):
    # check against onset threshold
    if ema >= ONSET_THRESH:
        # start transition timer
        if context.state_change_start_ts is None:
            context.state_change_start_ts = ts
            return None
        
        # calculate elapsed transition time
        elapsed = (ts - context.state_change_start_ts).total_seconds()
        
        # confirm sleep onset
        if elapsed >= ONSET_DURATION_SEC:
            context.currently_asleep = True
            context.sleep_onset_time = ts
            context.state_change_start_ts = None
            return "sleep_onset"
    else:
        # reset timer if threshold not met
        context.state_change_start_ts = None
    
    return None

def check_wake_criteria(context, ema, ts):
    # check against wake threshold
    if ema <= WAKE_THRESH:
        # start transition timer
        if context.state_change_start_ts is None:
            context.state_change_start_ts = ts
            return None
        
        # calculate elapsed transition time
        elapsed = (ts - context.state_change_start_ts).total_seconds()
        
        # confirm wake up
        if elapsed >= WAKE_DURATION_SEC:
            context.currently_asleep = False
            context.state_change_start_ts = None
            return "wake_up"
    else:
        # reset timer if threshold not met
        context.state_change_start_ts = None
    
    return None

def on_sleep_onset(context, ts):
    # log detection event
    print(f" >>> DETECTED SLEEP ONSET AT {ts}")
    
    db.insert_event(
        context.night_id,
        "sleep_onset",
        payload={"ts": str(ts), "confidence": getattr(context, 'prob_ema', 0.0)}
    )

    # retrieve calendar constraint
    hard_limit = logic.get_calendar_limit(context)

    # calculate debt recovery target
    sleep_debt_hours = logic.calculate_sleep_debt()
    ideal_hours = 8.0 + (sleep_debt_hours * 0.5) 
    
    ideal_wake_time = ts + timedelta(hours=ideal_hours)
    
    # ensure alarm respects hard constraints
    calendar_dt = datetime.combine(context.until.date(), hard_limit)
    final_alarm_dt = min(ideal_wake_time, calendar_dt)
    
    print(f"LOGIC: Onset {ts}. Debt {sleep_debt_hours:.2f}h. Ideal {ideal_wake_time}. Limit {calendar_dt}.")
    
    # update system alarm
    scheduler.scheduler.set_alarm(final_alarm_dt)

def on_wake(context, ts):
    # log detection event
    print(f" >>> DETECTED WAKE UP AT {ts}")
    
    db.insert_event(
        context.night_id,
        "wake_up",
        payload={"ts": str(ts), "confidence": getattr(context, 'prob_ema', 0.0)}
    )