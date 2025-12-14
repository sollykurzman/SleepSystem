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

@dataclass#(frozen=True)
class NightContext:
    cutoff: dt_time
    night_id: str
    until: datetime

    classification_history: deque
    currently_asleep: bool
    sleep_onset_time: datetime | None

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

    db.init_db()

    store_thread = threading.Thread(
        target=store.store_worker,
        args=(context.night_id, context.until),
        daemon=True
    )
    store_thread.start()
    # store.start_workers(context.night_id, context.until)

    listener_thread = threading.Thread(
        target=listener.listener,
        args=(context.until,),
        daemon=True
    )
    listener_thread.start()

    classifier_thread = threading.Thread(
        target=classifier.classification_worker,
        args=(context.night_id, context.until),
        daemon=True
    )
    classifier_thread.start()
    # classifier.start_classifier(context.night_id, context.until)

    action_thread = threading.Thread(
        target=action.action_loop,
        args=(context,),
        daemon=True
    )
    action_thread.start()

    print(f"Running until {context.until}. Press Ctrl+C to stop.")

    try:
        while True:
            if not (store_thread.is_alive() and listener_thread.is_alive()):
                break
            
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nCtrl+C received! Stopping...")
