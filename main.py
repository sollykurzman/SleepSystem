#!/usr/bin/python3

from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass
import threading
import time
from collections import deque

import listener
import store
import classifier
import action
import db
import ui
import scheduler
import util_logic
import util_hardware

#class used to store context for the night
@dataclass
class NightContext:
    cutoff: dt_time
    night_id: str
    until: datetime

    classification_history: deque
    currently_asleep: bool
    sleep_onset_time: datetime | None

#initiates the context for the current night
def build_night_context(cutoff):
    now = datetime.now()
    if now.time() < cutoff:
        now -= timedelta(days=1)

    night_id = now.strftime("%d%m%y")
    until = datetime.combine((now + timedelta(days=1)).date(), cutoff)

    return NightContext(
        cutoff=cutoff,
        night_id=night_id,
        until=until,
        classification_history=deque(maxlen=60),
        currently_asleep=False,
        sleep_onset_time=None
    )

if __name__ == "__main__":
    context = build_night_context(dt_time(14, 0))

    print(f"Running until {context.until} for night {context.night_id}")

    #initiate database
    db.init_db()

    #start alarm scheduler
    scheduler.start_scheduler()

    util_logic.determine_initial_alarm(context)

    #begin data store thread, stores and formats data
    store_thread = threading.Thread(
        target=store.store_worker,
        args=(context.night_id, context.until),
        daemon=True
    )
    store_thread.start()

    #begin listener thread, presents data to the store
    listener_thread = threading.Thread(
        target=listener.listener,
        args=(context.until,),
        daemon=True
    )
    listener_thread.start()

    #begin classifier thread, classifies the formatted data
    classifier_thread = threading.Thread(
        target=classifier.classification_worker,
        args=(context.night_id, context.until),
        daemon=True
    )
    classifier_thread.start()

    #begin action thread, actions the data
    action_thread = threading.Thread(
        target=action.action_loop,
        args=(context,),
        daemon=True
    )
    action_thread.start()

    #begin ui thread, shows real-time data
    ui.start_ui_thread()

    print(f"Running until {context.until}. Press Ctrl+C to stop.")

    #keep main thread alive while others run
    try:
        while datetime.now() < context.until:
            # Check if critical data threads have died unexpectedly
            if not (store_thread.is_alive() and listener_thread.is_alive()):
                print("Critical threads stopped unexpectedly.")
                break
            
            time.sleep(1)

        print("End time reached. Shutting down...")

    #exit on Ctrl+C
    except KeyboardInterrupt:
        print("\nCtrl+C received! Stopping...")

    finally:
        # Graceful Shutdown Sequence
        print("Cleaning up resources...")
        
        # Stop the scheduler loop
        if hasattr(scheduler.scheduler, 'running'):
            scheduler.scheduler.running = False
            
        # Stop any active hardware (Buzzer/Lights)
        util_hardware.hw.stop_all()
        
        print("Shutdown complete.")
