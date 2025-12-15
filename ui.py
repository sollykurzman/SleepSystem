#!/usr/bin/python3

import threading
from flask import Flask, render_template, jsonify, request
import sqlite3
import pandas as pd
from datetime import datetime
import json
import logging
import time

import db
import scheduler 
import util_hardware

app = Flask(__name__)

# disable flask request logging for cleaner console output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def get_db_data(query, args=()):
    try:
        # connect to database using shared path
        conn = sqlite3.connect(db.DB_PATH) 
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # execute query and fetch results
        cur.execute(query, args)
        rows = cur.fetchall()
        
        conn.close()
        return rows
    except Exception as e:
        print(f"UI DB Error: {e}")
        return []

@app.route('/')
def index():
    # serve main dashboard page
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    # fetch most recent classification record
    latest_cls = get_db_data("SELECT * FROM classifications ORDER BY id DESC LIMIT 1")
    current_state = latest_cls[0]['label'] if latest_cls else "Unknown"
    last_update = latest_cls[0]['ts'] if latest_cls else None

    # identify current night session
    night_row = get_db_data("SELECT DISTINCT night_id FROM classifications ORDER BY id DESC LIMIT 1")
    night_id = night_row[0]['night_id'] if night_row else "N/A"
    
    # retrieve sleep events for current night
    events = get_db_data("SELECT * FROM sleep_events WHERE night_id=? ORDER BY ts DESC", (night_id,))
    event_list = [{'time': e['ts'], 'type': e['type'], 'payload': e['payload']} for e in events]

    # check for active alarms
    alarm = get_db_data("SELECT * FROM alarms WHERE active=1 LIMIT 1")
    alarm_time = alarm[0]['target_ts'] if alarm else "No Alarm"

    return jsonify({
        "night_id": night_id,
        "state": current_state,
        "last_update": last_update,
        "alarm_time": alarm_time,
        "events": event_list
    })

@app.route('/api/history')
def api_history():
    # fetch list of recent night session ids
    nights = get_db_data("SELECT DISTINCT night_id FROM classifications ORDER BY id DESC LIMIT 10")
    return jsonify([n['night_id'] for n in nights])

@app.route('/api/graph/<night_id>')
def api_graph(night_id):
    # resolve current night id if requested
    if night_id == "current":
        row = get_db_data("SELECT DISTINCT night_id FROM classifications ORDER BY id DESC LIMIT 1")
        if not row:
            return jsonify({"error": "No data found"})
        night_id = row[0]['night_id']

    # retrieve full classification history for night
    rows = get_db_data(
        "SELECT ts, label, probabilities FROM classifications WHERE night_id=? ORDER BY ts ASC", 
        (night_id,)
    )

    timestamps = []
    labels = []
    probs = {} 
    
    # process records for frontend graphing
    for r in rows:
        timestamps.append(r['ts'])
        labels.append(r['label'])
        
        try:
            # parse nested probability data
            p_dict = json.loads(r['probabilities'])
            flat = {}
            
            # flatten probability structure
            if 'state' in p_dict and isinstance(p_dict['state'], dict):
                for k, v in p_dict['state'].items(): flat[k] = v
            if 'asleep' in p_dict and isinstance(p_dict['asleep'], dict):
                 for k, v in p_dict['asleep'].items(): flat[k] = v
            if 'in_bed' in p_dict and isinstance(p_dict['in_bed'], dict):
                 for k, v in p_dict['in_bed'].items(): flat[k] = v
            
            # append values to respective columns
            for k, v in flat.items():
                if k not in probs: probs[k] = []
                probs[k].append(v)
                
            # ensure all probability arrays remain aligned
            for k in probs:
                if k not in flat: probs[k].append(None)
                
        except:
            pass

    return jsonify({
        "night_id": night_id,
        "timestamps": timestamps,
        "labels": labels,
        "probabilities": probs
    })

@app.route('/api/alarm/set', methods=['POST'])
def set_alarm():
    """Sets a new alarm based on HH:MM input."""
    data = request.json
    time_str = data.get('time') # Expecting "HH:MM"
    
    if not time_str:
        return jsonify({"error": "No time provided"}), 400

    try:
        # Parse the incoming time
        target_time = datetime.strptime(time_str, "%H:%M").time()
        now = datetime.now()
        target_dt = datetime.combine(now.date(), target_time)

        # Smart Scheduling: If the time has passed today, assume tomorrow
        if target_dt <= now:
            target_dt += timedelta(days=1)

        # Update the system scheduler
        scheduler.scheduler.set_alarm(target_dt)
        
        return jsonify({
            "status": "success", 
            "target": target_dt.strftime("%Y-%m-%d %H:%M:%S")
        })
    except ValueError:
        return jsonify({"error": "Invalid time format"}), 400

@app.route('/api/alarm/cancel', methods=['POST'])
def cancel_alarm():
    scheduler.scheduler.cancel_alarm()
    util_hardware.hw.stop_all()
    return jsonify({"status": "cancelled"})

@app.route('/api/alarm/snooze', methods=['POST'])
def snooze_alarm():
    # Stop buzzer, reset alarm for +9 mins
    util_hardware.hw.stop_all()
    new_time = datetime.now() + timedelta(minutes=9)
    scheduler.scheduler.set_alarm(new_time, fade_minutes=0)
    return jsonify({"status": "snoozed", "new_time": new_time})

def run_server():
    # start flask server on port 5001
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

def start_ui_thread():
    # launch ui server in background thread
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    print("UI Server started at http://localhost:5001")

if __name__ == "__main__":
    start_ui_thread()
    # keep main thread alive for testing
    while True: time.sleep(1)