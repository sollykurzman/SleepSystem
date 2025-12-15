#!/usr/bin/python3

import threading
import time
from datetime import datetime, timedelta
import pandas as pd

import db
import hardware

class RobustScheduler:
    def __init__(self):
        # initialize threading lock and state flags
        self.lock = threading.Lock()
        self.running = True
        self.alarm_active = False
        self.target_time = None
        self.fade_started = False
        
        # restore state from persistent storage
        self._load_state()

    def _load_state(self):
        # recover active alarm settings from database
        row = db.get_active_alarm() 
        if row:
            # parse stored alarm data
            target = pd.to_datetime(row['target_ts']).to_pydatetime()
            if row['active'] == 1 and row['fired'] == 0:
                print(f"RECOVERY: Found active alarm for {target}")
                # restore alarm without duplicating db entry
                self.set_alarm(target, update_db=False)
    
    def get_schedule(self):
        #get the current alarm time with lock
        with self.lock:
            return self.target_time

    def set_alarm(self, target_dt, update_db=True):
        with self.lock:
            # update internal state
            self.target_time = target_dt
            self.alarm_active = True
            self.fade_started = False
            
            if update_db:
                # persist new alarm to database
                db.insert_alarm(target_dt, "scheduler", "set_alarm", {})
                
                # estimate night id based on target time
                night_id_est = (target_dt - timedelta(hours=12)).strftime("%d%m%y")
                
                # log target time as an event for historical tracking
                db.insert_event(
                    night_id_est, 
                    "alarm_target", 
                    ts=target_dt  
                )
            
            print(f"SCHEDULER: Alarm set for {self.target_time}")

    def cancel_alarm(self):
        with self.lock:
            # clear internal alarm state
            self.alarm_active = False
            self.target_time = None
            
            # update database status
            db.update_alarm_status(None, active=0)
            
            # stop any active hardware effects
            hardware.hw.stop_all()
            print("SCHEDULER: Alarm cancelled.")

    def run(self):
        print("SCHEDULER: Monitoring started.")
        while self.running:
            try:
                # check for events every second
                time.sleep(1) 
                
                with self.lock:
                    if not self.alarm_active or not self.target_time:
                        continue

                    now = datetime.now()
                    
                    # check if light fade should begin
                    fade_start_time = self.target_time - timedelta(minutes=30)
                    if now >= fade_start_time and not self.fade_started:
                        print("SCHEDULER: Triggering Light Fade")
                        # calculate remaining duration for fade
                        duration = (self.target_time - now).total_seconds()
                        if duration > 0:
                            hardware.hw.trigger_fade(duration=duration)
                        self.fade_started = True

                    # check if alarm should trigger
                    if now >= self.target_time:
                        print("SCHEDULER: Triggering ALARM!")
                        hardware.hw.trigger_alarm()
                        
                        # update alarm status in database
                        conn = db.get_conn()
                        conn.execute("UPDATE alarms SET fired=1, active=0 WHERE active=1")
                        conn.commit()
                        conn.close()
                        
                        # disable alarm locally
                        self.alarm_active = False 
                        
            except Exception as e:
                print(f"SCHEDULER ERROR: {e}")

# create singleton instance
scheduler = RobustScheduler()

def start_scheduler():
    # launch scheduler loop in background thread
    t = threading.Thread(target=scheduler.run, daemon=True)
    t.start()