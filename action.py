import threading
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass

import classifier
import db
import alarm_scheduler


# ------------------ Engine ------------------

def action_loop(context):
    db.insert_event(context.night_id, "sleep_engine_started")
    while True:
        result = classifier.classification_queue.get()

        handle_classification(
            context = context,
            ts=result["datetime"],
            label=result["classification"],
            probabilities=result["probabilities"]
        )


# ------------------ Core Handling ------------------

def handle_classification(context, ts, label, probabilities):
    context.classification_history.append(label)

    db.insert_classification(
        context.night_id,
        ts,
        label,
        probabilities
    )

    evaluate_sleep_state(context, ts)


# ------------------ Sleep State ------------------

def evaluate_sleep_state(context, ts):

    if not context.classification_history:
        return

    sleep_votes = sum(
        1 for s in context.classification_history
        if s in ("Core Sleep", "REM", "Deep Sleep")
    )

    total = len(context.classification_history)
    if total == 0:
        return

    ratio = sleep_votes / total
    if not context.currently_asleep and ratio >= 0.75:
        context.currently_asleep = True
        context.sleep_onset_time = ts
        on_sleep_onset(context, ts)

    elif context.currently_asleep and ratio <= 0.40:
        context.currently_asleep = False
        on_wake(context, ts)

# ------------------ Decision Hooks ------------------

def on_sleep_onset(context, ts):
    db.insert_event(
        context.night_id,
        "sleep_onset",
        payload={"ts": str(ts)}
    )

    # decide_alarm_from_calendar(context, ts)
    # decide_alarm_from_sleep_debt(context, ts)


def on_wake(context, ts):
    db.insert_event(
        context.night_id,
        "wake_up",
        payload={"ts": str(ts)}
    )
