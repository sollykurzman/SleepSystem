#!/usr/bin/python3

import time
import os
import threading
from datetime import datetime
from collections import deque
import queue
import pandas as pd
import joblib
import numpy as np

import store
from util_format import process_window, add_history_features
# CHANGED: Import the advanced feature generator
# Ensure util_train2 has add_advanced_features from the latest training script
from util_train2gem import add_advanced_features 

# ---------------- CONFIG ----------------

ML_INTERVAL = 2.0
MODEL_PATH = "ML/"
WINDOW_SECONDS = 30
# CHANGED: Increased buffer size to support the 60-sample long window
BUFFER_SIZE = 60 

# ---------------- LOAD MODELS ----------------

in_bed_model = joblib.load(MODEL_PATH + "in_bed_model.joblib")
in_bed_encoder = joblib.load(MODEL_PATH + "in_bed_encoder.joblib")

asleep_model = joblib.load(MODEL_PATH + "asleep_model.joblib")
asleep_encoder = joblib.load(MODEL_PATH + "asleep_encoder.joblib")

state_model = joblib.load(MODEL_PATH + "state_model.joblib")
state_encoder = joblib.load(MODEL_PATH + "state_encoder.joblib")

# Refined Model
try:
    state_2_model = joblib.load(MODEL_PATH + "state_2_model.joblib")
    state_2_features = joblib.load(MODEL_PATH + "state_2_features.joblib")
except Exception as e:
    print(f"Warning: Could not load refined model ({e}). Ensure USE_REFINED_MODEL is False.")
    state_2_model = None
    state_2_features = []


# ---------------- HISTORY BUFFERS ----------------

class HistoryBuffer:
    def __init__(self, max_length):
        self.buffer = deque(maxlen=max_length)
        self.lock = threading.Lock()

    def add(self, x):
        with self.lock:
            self.buffer.append(x)

    def get(self):
        with self.lock:
            return list(self.buffer)

history_buffer = HistoryBuffer(max_length=12)

# Buffer for the Refined Model (Stores the probability dictionaries)
# CHANGED: Use BUFFER_SIZE (60)
probability_buffer = deque(maxlen=BUFFER_SIZE)

classification_queue = queue.Queue()

# ---------------- ML HELPERS ----------------

def predict_with_model(model, encoder, X):
    required = model.feature_names_in_
    X = X[required]

    label_idx = model.predict(X)[0]
    label = encoder.inverse_transform([label_idx])[0]

    proba = model.predict_proba(X)[0]
    classes = encoder.inverse_transform(model.classes_)

    probabilities = {
        cls: float(p) for cls, p in zip(classes, proba)
    }

    return label, probabilities

def flatten_probabilities(ib_proba, slp_proba, state_proba):
    """Flatten nested dicts for the refined model input."""
    flat = {}
    flat['in_bed__inBed'] = ib_proba.get('inBed', 0.0)
    flat['asleep__Asleep'] = slp_proba.get('Asleep', 0.0)
    flat['asleep__Awake'] = slp_proba.get('Awake', 0.0)
    flat['state__Core Sleep'] = state_proba.get('Core Sleep', 0.0)
    flat['state__Deep Sleep'] = state_proba.get('Deep Sleep', 0.0)
    
    # Handle naming differences (REM vs REM Sleep)
    rem_val = state_proba.get('REM Sleep', state_proba.get('REM', 0.0))
    flat['state__REM Sleep'] = rem_val # Updated to match training script key
    return flat

def classify_snippet(snippet, use_refined=True):
    X = snippet.drop(columns=["timestamp"])

    # 1. Run Base Models (Always required)
    ib_label, ib_proba = predict_with_model(in_bed_model, in_bed_encoder, X)
    slp_label, slp_proba = predict_with_model(asleep_model, asleep_encoder, X)
    state_label_base, state_proba_base = predict_with_model(state_model, state_encoder, X)

    # 2. Update Probability Buffer (Always do this to keep history warm)
    flat_probs = flatten_probabilities(ib_proba, slp_proba, state_proba_base)
    probability_buffer.append(flat_probs)

    # 3. Decision Logic
    
    # Case A: Not In Bed
    if ib_label != "inBed":
        return ib_label, {"in_bed": ib_proba}

    # Case B: Awake
    if slp_label != "Asleep":
        return slp_label, {
            "in_bed": ib_proba,
            "asleep": slp_proba
        }

    # Case C: Asleep
    
    # OPTION 1: Use Refined Model (Improved Accuracy)
    if use_refined and state_2_model is not None:
        try:
            # Convert buffer to DataFrame for advanced feature calc
            df_probs = pd.DataFrame(list(probability_buffer))
            
            # CHANGED: Use add_advanced_features instead of add_rolling_features
            df_refined, _ = add_advanced_features(df_probs)
            
            # Prepare input vector (ensure columns match training)
            # Reindex fills missing columns with 0 automatically
            X_refined = df_refined.iloc[[-1]].reindex(columns=state_2_features, fill_value=0)
            
            final_label = state_2_model.predict(X_refined)[0]
            final_proba_array = state_2_model.predict_proba(X_refined)[0]
            
            state_2_probs_dict = {cls: float(p) for cls, p in zip(state_2_model.classes_, final_proba_array)}
            
            return final_label, {
                "in_bed": ib_proba,
                "asleep": slp_proba,
                "state": state_2_probs_dict # Refined probabilities
            }
        except Exception as e:
            # Fallback silently to base model if buffer not full enough or error
            pass

    # OPTION 2: Use Base Model (Training Data Generation or Fallback)
    return state_label_base, {
        "in_bed": ib_proba,
        "asleep": slp_proba,
        "state": state_proba_base # Raw base probabilities
    }


def write_classification_row(night_id, result, csv_path=None):
    if csv_path is None:
        csv_path = f"Data/{night_id}/classification-{night_id}.csv"

    ALL_COLUMNS = {
        "in_bed": ["inBed", "notInBed"],
        "asleep": ["Asleep", "Awake"],
        "state": ["Core Sleep", "REM Sleep", "Deep Sleep"]
    }

    row = {
        "datetime": result["datetime"],
        "classification": result["classification"]
    }

    for stage, classes in ALL_COLUMNS.items():
        probs = result["probabilities"].get(stage, {})
        for cls in classes:
            row[f"{stage}__{cls}"] = probs.get(cls, None)

    df_out = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)

    df_out.to_csv(csv_path, mode="a", header=write_header, index=False)


def classify_once(night_id=None, write=True, timestamp=None, use_refined=True):
    times, volts = store.live_buffer.get_snapshot()
    if times is None:
        return None

    df = pd.DataFrame({
        "datetime": times,
        "voltage": volts
    })

    snippet = process_window(
        df["datetime"].values,
        df["voltage"].values,
        df["datetime"].iloc[0],
        WINDOW_SECONDS
    )

    if snippet is None:
        return None

    # Update Raw History
    history = history_buffer.get()
    history_buffer.add(snippet)

    snippet = pd.DataFrame(history + [snippet])
    snippet = snippet.drop(columns=["sleep_state"], errors="ignore")
    snippet = add_history_features(snippet)

    # Classify (Will switch logic based on USE_REFINED_MODEL)
    label, probabilities = classify_snippet(snippet, use_refined=use_refined)

    if timestamp is None:
        timestamp = datetime.now()

    result = {
        "datetime": timestamp,
        "classification": label,
        "probabilities": probabilities
    }

    classification_queue.put(result)

    if night_id and write:
        write_classification_row(night_id, result)

    return result


# ---------------- CLASSIFICATION WORKER ----------------

def classification_worker(night_id=None, until=None, use_refined=True):
    print("Classification worker started")

    while until is None or datetime.now() < until:
        time.sleep(ML_INTERVAL)
        classify_once(night_id, use_refined=use_refined)


# ---------------- PUBLIC API ----------------

def start_classifier(night_id=None, until=None):
    t = threading.Thread(
        target=classification_worker,
        args=(night_id, until),
        daemon=True
    )
    t.start()